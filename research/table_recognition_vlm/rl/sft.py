"""LoRA SFT warm-up for image-to-table-HTML generation before GRPO."""

import argparse
import inspect
import json
import os
from io import BytesIO
from pathlib import Path

from grpo import (
    DEFAULT_DATASET_ID,
    DEFAULT_MODEL_NAME,
    DEFAULT_PROMPT_FILE,
    load_datasets,
    load_prompt,
    patch_qwen35_training_linear_attention,
)
from reward import parse_table_html


DEFAULT_OUTPUT_DIR = "chandra_ocr_2_table_html_sft"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--dataset-revision", default=None)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prompt-file", default=str(DEFAULT_PROMPT_FILE))
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=64)
    parser.add_argument("--include-invalid", action="store_true")

    parser.add_argument("--max-seq-length", type=int, default=20_480)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--report-to", default="none")

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--language-only", action="store_true")
    parser.add_argument("--load-in-16bit", action="store_true")
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and verify data without loading the model.",
    )
    return parser.parse_args(argv)


def validate_args(args):
    positive = (
        "max_seq_length",
        "num_train_epochs",
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "logging_steps",
        "lora_r",
        "lora_alpha",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in ("max_train_samples", "max_eval_samples"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive when provided")
    if args.max_steps == 0 or args.max_steps < -1:
        raise ValueError("--max-steps must be -1 or positive")
    if not 0 <= args.warmup_ratio < 1:
        raise ValueError("--warmup-ratio must be in [0, 1)")
    if not 0 <= args.lora_dropout < 1:
        raise ValueError("--lora-dropout must be in [0, 1)")
    if args.resume_from_checkpoint:
        checkpoint = Path(args.resume_from_checkpoint).expanduser()
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint}")
        args.resume_from_checkpoint = str(checkpoint)


def to_rgb(image):
    from PIL import Image

    if isinstance(image, Image.Image):
        return image if image.mode == "RGB" else image.convert("RGB")
    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            return opened.convert("RGB")
    if isinstance(image, bytes):
        with Image.open(BytesIO(image)) as opened:
            return opened.convert("RGB")
    if isinstance(image, dict):
        for key in ("bytes", "path", "image"):
            if image.get(key) is not None:
                return to_rgb(image[key])
    raise TypeError(f"Unsupported image type: {type(image).__name__}")


def to_conversation(sample, prompt):
    user_content = [{"type": "text", "text": prompt}]
    user_content.extend(
        {"type": "image", "image": to_rgb(image)} for image in sample["images"]
    )
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["table_html"]}],
            },
        ]
    }


class ConversationDataset:
    """Lazily materialize images so the full image corpus is not held in RAM."""

    def __init__(self, dataset, prompt):
        self.dataset = dataset
        self.prompt = prompt

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return to_conversation(self.dataset[index], self.prompt)


def make_config(args, bfloat16_supported, eos_token):
    from trl import SFTConfig

    kwargs = {
        "output_dir": args.output_dir,
        "max_steps": args.max_steps,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "lr_scheduler_type": "cosine",
        "optim": "adamw_8bit",
        "bf16": bfloat16_supported,
        "fp16": not bfloat16_supported,
        "logging_steps": args.logging_steps,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "seed": args.seed,
        "data_seed": args.seed,
        "report_to": args.report_to,
        "remove_unused_columns": False,
        "dataset_text_field": "",
        "dataset_kwargs": {"skip_prepare_dataset": True},
        "packing": False,
    }
    if eos_token is not None:
        kwargs["eos_token"] = eos_token
    parameters = inspect.signature(SFTConfig).parameters
    kwargs["max_length" if "max_length" in parameters else "max_seq_length"] = (
        args.max_seq_length
    )
    return SFTConfig(**kwargs)


def print_summary(train_dataset, eval_dataset, args):
    sample = train_dataset[0]
    parsed = parse_table_html(sample["table_html"])
    if not parsed.valid:
        raise ValueError("The first prepared target is not complete table HTML")
    print(
        json.dumps(
            {
                "stage": "sft",
                "dataset": args.dataset_id,
                "model": args.model_name,
                "output_dir": args.output_dir,
                "train_rows": len(train_dataset),
                "eval_rows": len(eval_dataset),
                "sample_images": len(sample["images"]),
                "sample_shape": [parsed.num_rows, parsed.num_cols],
                "reasoning_in_target": False,
            },
            indent=2,
        )
    )


def main(argv=None):
    args = parse_args(argv)
    validate_args(args)
    if not args.dry_run:
        os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
        # Unsloth must patch Torch, Transformers, and TRL before they are imported.
        import unsloth  # noqa: F401

    prompt = load_prompt(args.prompt_file)
    train_raw, eval_raw = load_datasets(args, prompt)
    print_summary(train_raw, eval_raw, args)
    if args.dry_run:
        sample = to_conversation(train_raw[0], prompt)
        target = sample["messages"][1]["content"][0]["text"]
        print(f"Dry run passed: {len(target)} HTML target characters.")
        return

    from trl import SFTTrainer
    from unsloth import FastVisionModel, is_bfloat16_supported
    from unsloth.trainer import UnslothVisionDataCollator

    model, processor = FastVisionModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=not args.load_in_16bit,
        load_in_16bit=args.load_in_16bit,
        use_gradient_checkpointing="unsloth",
        max_lora_rank=args.lora_r,
    )
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=not args.language_only,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=None,
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
    )
    patch_qwen35_training_linear_attention(model)
    FastVisionModel.for_training(model)

    train_dataset = ConversationDataset(train_raw, prompt)
    eval_dataset = ConversationDataset(eval_raw, prompt)
    eos_token = getattr(processor, "eos_token", None)
    if eos_token is None and hasattr(processor, "tokenizer"):
        eos_token = processor.tokenizer.eos_token

    trainer_kwargs = {
        "model": model,
        "args": make_config(args, is_bfloat16_supported(), eos_token),
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": UnslothVisionDataCollator(
            model,
            processor,
            train_on_responses_only=True,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
        ),
    }
    if "processing_class" in inspect.signature(SFTTrainer).parameters:
        trainer_kwargs["processing_class"] = processor
    else:
        trainer_kwargs["tokenizer"] = processor

    trainer = SFTTrainer(**trainer_kwargs)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
