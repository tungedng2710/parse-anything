import argparse
import inspect
import json
import os
from io import BytesIO
from pathlib import Path


DEFAULT_DATASET_ID = "tungedng2710/table_html_with_reasoning"
DEFAULT_MODEL_NAME = "Qwen/Qwen3.5-2B"
DEFAULT_OUTPUT_DIR = "qwen35_2b_table_html_lora"

PROMPT = (
    "Convert the table in the provided image into structural HTML. Preserve all visible "
    "text, empty cells, row and column order, and rowspan and colspan attributes. If there "
    "are multiple images, they are consecutive parts of the same table. Return only the "
    "complete <table>...</table> markup, with no reasoning, explanation, or markdown."
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen3.5-2B with Unsloth for image-to-table-HTML generation."
    )
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--dataset-revision", default=None)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default=None)
    parser.add_argument("--eval-size", type=float, default=0.05)
    parser.add_argument("--max-seq-length", type=int, default=20_480)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)

    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--dataset-num-proc", type=int, default=1)
    parser.add_argument("--seed", type=int, default=3407)

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument(
        "--language-only",
        action="store_true",
        help="Do not add LoRA adapters to vision layers.",
    )
    parser.add_argument(
        "--load-in-16bit",
        action="store_true",
        help="Load the base model in 16-bit instead of the default 4-bit QLoRA mode.",
    )
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Download, split, and validate the dataset without loading the model.",
    )
    return parser.parse_args(argv)


def validate_args(args):
    if args.eval_split is None and not 0 < args.eval_size < 1:
        raise ValueError("--eval-size must be between 0 and 1.")
    positive_arguments = (
        "max_seq_length",
        "num_train_epochs",
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "logging_steps",
        "dataset_num_proc",
        "lora_r",
        "lora_alpha",
    )
    for name in positive_arguments:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    for name in ("max_train_samples", "max_eval_samples"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive when provided.")
    if args.lora_dropout < 0 or args.lora_dropout >= 1:
        raise ValueError("--lora-dropout must be in [0, 1).")
    if args.resume_from_checkpoint is not None:
        checkpoint = Path(args.resume_from_checkpoint).expanduser()
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint}")
        args.resume_from_checkpoint = str(checkpoint)


def extract_table_html(value):
    if not isinstance(value, str):
        raise TypeError(f"table_html must be a string, got {type(value).__name__}")

    lowercase = value.lower()
    reasoning_end_tag = "</think>"
    reasoning_end = lowercase.rfind(reasoning_end_tag)
    search_start = reasoning_end + len(reasoning_end_tag) if reasoning_end >= 0 else 0
    start = lowercase.find("<table", search_start)
    end_tag = "</table>"
    end = lowercase.rfind(end_tag)
    if start < 0 or end < start:
        raise ValueError("table_html does not contain a complete <table>...</table> element")
    return value[start : end + len(end_tag)].strip()


def load_datasets(args):
    from datasets import load_dataset

    load_kwargs = {}
    if args.dataset_revision:
        load_kwargs["revision"] = args.dataset_revision
    dataset = load_dataset(args.dataset_id, args.dataset_config, **load_kwargs)

    if args.train_split not in dataset:
        raise ValueError(
            f"Train split {args.train_split!r} was not found; available splits: {list(dataset)}"
        )
    train_dataset = dataset[args.train_split]
    if args.eval_split is not None:
        if args.eval_split not in dataset:
            raise ValueError(
                f"Eval split {args.eval_split!r} was not found; available splits: {list(dataset)}"
            )
        eval_dataset = dataset[args.eval_split]
    else:
        split = train_dataset.train_test_split(test_size=args.eval_size, seed=args.seed)
        train_dataset, eval_dataset = split["train"], split["test"]

    if args.max_train_samples is not None:
        train_dataset = train_dataset.select(
            range(min(args.max_train_samples, len(train_dataset)))
        )
    if args.max_eval_samples is not None:
        eval_dataset = eval_dataset.select(range(min(args.max_eval_samples, len(eval_dataset))))
    if len(train_dataset) == 0 or len(eval_dataset) == 0:
        raise ValueError("Both training and evaluation datasets must contain at least one row.")

    validate_dataset(train_dataset, "train")
    validate_dataset(eval_dataset, "eval")
    return train_dataset, eval_dataset


def validate_dataset(dataset, split_name):
    required_columns = {"images", "table_html"}
    missing = required_columns.difference(dataset.column_names)
    if missing:
        raise ValueError(f"{split_name} split is missing columns: {sorted(missing)}")

    errors = []
    for index, sample in enumerate(dataset):
        if not sample["images"]:
            errors.append(f"row {index}: images is empty")
        try:
            extract_table_html(sample["table_html"])
        except (TypeError, ValueError) as error:
            errors.append(f"row {index}: {error}")
        if len(errors) == 10:
            break
    if errors:
        raise ValueError(f"Invalid {split_name} split:\n" + "\n".join(errors))


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


def to_conversation(sample):
    user_content = [{"type": "text", "text": PROMPT}]
    user_content.extend(
        {"type": "image", "image": to_rgb(image)} for image in sample["images"]
    )
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": extract_table_html(sample["table_html"])}
                ],
            },
        ]
    }


def make_sft_config(args, bfloat16_supported, eos_token):
    from trl import SFTConfig

    config_kwargs = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "lr_scheduler_type": "linear",
        "optim": "adamw_8bit",
        "fp16": not bfloat16_supported,
        "bf16": bfloat16_supported,
        "logging_steps": args.logging_steps,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "save_total_limit": 2,
        "prediction_loss_only": True,
        "seed": args.seed,
        "report_to": "none",
        "remove_unused_columns": False,
        "dataset_text_field": "",
        "dataset_kwargs": {"skip_prepare_dataset": True},
        "dataset_num_proc": args.dataset_num_proc,
        "packing": False,
    }
    if eos_token is not None:
        config_kwargs["eos_token"] = eos_token

    parameters = inspect.signature(SFTConfig).parameters
    length_name = "max_length" if "max_length" in parameters else "max_seq_length"
    config_kwargs[length_name] = args.max_seq_length
    return SFTConfig(**config_kwargs)


def print_summary(train_dataset, eval_dataset, args):
    targets = [extract_table_html(value) for value in train_dataset["table_html"]]
    targets.extend(extract_table_html(value) for value in eval_dataset["table_html"])
    lengths = sorted(map(len, targets))
    print(
        json.dumps(
            {
                "dataset": args.dataset_id,
                "model": args.model_name,
                "train_rows": len(train_dataset),
                "eval_rows": len(eval_dataset),
                "html_target_characters": {
                    "min": lengths[0],
                    "median": lengths[len(lengths) // 2],
                    "max": lengths[-1],
                },
                "reasoning_in_target": False,
            },
            indent=2,
        )
    )


def main(argv=None):
    args = parse_args(argv)
    validate_args(args)

    if not args.dry_run:
        # Unsloth 2026.6.9 generates a broken Qwen3.5 linear-attention wrapper.
        os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
        # Unsloth must be imported before Transformers, TRL, and Torch.
        import unsloth  # noqa: F401

    train_raw, eval_raw = load_datasets(args)
    print_summary(train_raw, eval_raw, args)

    if args.dry_run:
        sample = to_conversation(train_raw[0])
        target = sample["messages"][1]["content"][0]["text"]
        image_count = len(sample["messages"][0]["content"]) - 1
        print(
            f"Dry run passed: {image_count} image(s), "
            f"{len(target)} HTML target characters."
        )
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
        target_modules="all-linear",
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
    )
    FastVisionModel.for_training(model)

    train_dataset = [to_conversation(sample) for sample in train_raw]
    eval_dataset = [to_conversation(sample) for sample in eval_raw]
    eos_token = getattr(processor, "eos_token", None)
    if eos_token is None and hasattr(processor, "tokenizer"):
        eos_token = processor.tokenizer.eos_token

    trainer_kwargs = {
        "model": model,
        "args": make_sft_config(args, is_bfloat16_supported(), eos_token),
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

    # load_best_model_at_end makes the adapter at output_dir the best one. The two
    # checkpoint-* directories are the best and latest resumable checkpoints.
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
