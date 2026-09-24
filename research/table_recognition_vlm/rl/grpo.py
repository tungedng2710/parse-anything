"""RLVR fine-tuning for image-to-table-HTML generation."""

import argparse
import inspect
import json
import os
from pathlib import Path

from reward import REWARD_FUNCTIONS, REWARD_WEIGHTS, extract_table_html, parse_table_html


DEFAULT_DATASET_ID = "tungedng2710/table_html_with_reasoning"
DEFAULT_MODEL_NAME = "datalab-to/chandra-ocr-2"
DEFAULT_OUTPUT_DIR = "chandra_ocr_2_table_html_grpo"
DEFAULT_PROMPT_FILE = Path(__file__).with_name("prompt.md")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--dataset-revision", default=None)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--prompt-file",
        default=str(DEFAULT_PROMPT_FILE),
        help="UTF-8 text file containing the user instruction.",
    )
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=64)
    parser.add_argument("--include-invalid", action="store_true")

    parser.add_argument("--max-seq-length", type=int, default=20_480)
    parser.add_argument("--max-prompt-length", type=int, default=1_024)
    parser.add_argument("--max-completion-length", type=int, default=8_192)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--num-iterations", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--beta", type=float, default=0.001)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--log-completions", action="store_true")

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--language-only", action="store_true")
    parser.add_argument("--load-in-16bit", action="store_true")
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load, filter, and verify the dataset without loading the model.",
    )
    return parser.parse_args(argv)


def validate_args(args):
    positive = (
        "max_seq_length",
        "max_prompt_length",
        "max_completion_length",
        "num_train_epochs",
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "logging_steps",
        "eval_steps",
        "save_steps",
        "lora_r",
        "lora_alpha",
        "num_iterations",
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
    if args.num_generations < 2:
        raise ValueError("--num-generations must be at least 2 for group-relative advantages")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    generation_batch = (
        args.per_device_train_batch_size * args.gradient_accumulation_steps * world_size
    )
    if generation_batch % args.num_generations:
        raise ValueError(
            "per-device batch size * gradient accumulation * world size must be divisible "
            "by --num-generations"
        )
    eval_batch = args.per_device_eval_batch_size * world_size
    if eval_batch % args.num_generations:
        raise ValueError(
            "per-device eval batch size * world size must be divisible by --num-generations"
        )
    if args.max_prompt_length + args.max_completion_length > args.max_seq_length:
        raise ValueError(
            "--max-prompt-length + --max-completion-length must not exceed --max-seq-length"
        )
    if not 0 < args.top_p <= 1 or args.temperature <= 0:
        raise ValueError("--top-p must be in (0, 1] and --temperature must be positive")
    if not 0 <= args.warmup_ratio < 1:
        raise ValueError("--warmup-ratio must be in [0, 1)")
    if args.beta < 0 or args.epsilon < 0:
        raise ValueError("--beta and --epsilon must be non-negative")
    if not 0 <= args.lora_dropout < 1:
        raise ValueError("--lora-dropout must be in [0, 1)")
    if args.resume_from_checkpoint:
        checkpoint = Path(args.resume_from_checkpoint).expanduser()
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint}")
        args.resume_from_checkpoint = str(checkpoint)


def load_prompt(prompt_file):
    path = Path(prompt_file).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Prompt file does not exist: {path}")
    prompt = path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"Prompt file is empty: {path}")
    return prompt


def max_lora_rank_for_model(model_name, requested_rank):
    """Include a local PEFT adapter's rank in Unsloth's allocation ceiling."""
    config_path = Path(model_name).expanduser() / "adapter_config.json"
    if not config_path.is_file():
        return requested_rank
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        ranks = [requested_rank, int(config.get("r", requested_rank))]
        ranks.extend(int(rank) for rank in config.get("rank_pattern", {}).values())
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid PEFT adapter config: {config_path}") from error
    return max(ranks)


def _training_record(_sample_id, table_html, prompt):
    return {
        "prompt": [{"role": "user", "content": prompt}],
        "table_html": extract_table_html(table_html),
    }


def prepare_split(dataset, args, split_name, max_samples, prompt):
    required = {
        "id",
        "images",
        "table_html",
        "reasoning",
        "num_rows",
        "num_cols",
        "num_cells",
        "has_merged_cells",
        "validation_passed",
        "num_images",
    }
    missing = required.difference(dataset.column_names)
    if missing:
        raise ValueError(f"{split_name} split is missing columns: {sorted(missing)}")
    if not args.include_invalid:
        dataset = dataset.filter(
            lambda passed: passed,
            input_columns=["validation_passed"],
            desc=f"Filtering invalid {split_name} annotations",
        )
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    if not len(dataset):
        raise ValueError(f"{split_name} split contains no usable rows")

    # Keep compact trace-derived columns for verification, but not the 100+ KB raw trace.
    dataset = dataset.remove_columns("reasoning")
    dataset = dataset.map(
        _training_record,
        input_columns=["id", "table_html"],
        fn_kwargs={"prompt": prompt},
        desc=f"Formatting {split_name} prompts",
    )
    return dataset.remove_columns("id")


def load_datasets(args, prompt):
    from datasets import load_dataset

    kwargs = {}
    if args.dataset_revision:
        kwargs["revision"] = args.dataset_revision
    datasets = load_dataset(args.dataset_id, args.dataset_config, **kwargs)
    for split in (args.train_split, args.eval_split):
        if split not in datasets:
            raise ValueError(f"Split {split!r} not found; available splits: {list(datasets)}")
    train_dataset = prepare_split(
        datasets[args.train_split], args, "train", args.max_train_samples, prompt
    )
    eval_dataset = prepare_split(
        datasets[args.eval_split], args, "eval", args.max_eval_samples, prompt
    )
    return train_dataset, eval_dataset


def print_summary(train_dataset, eval_dataset, args):
    sample = train_dataset[0]
    parsed = parse_table_html(sample["table_html"])
    if not parsed.valid:
        raise ValueError("The first prepared reference is not strict, complete table HTML")
    if len(sample["images"]) != sample["num_images"]:
        raise ValueError("The first prepared row has inconsistent image counts")
    print(
        json.dumps(
            {
                "dataset": args.dataset_id,
                "model": args.model_name,
                "train_rows": len(train_dataset),
                "eval_rows": len(eval_dataset),
                "sample_images": len(sample["images"]),
                "sample_shape": [parsed.num_rows, parsed.num_cols],
                "sample_cells": parsed.num_cells,
                "reasoning_in_prompt_or_completion": False,
                "reward_functions": [function.__name__ for function in REWARD_FUNCTIONS],
                "reward_weights": REWARD_WEIGHTS,
            },
            indent=2,
        )
    )


def make_config(args, bfloat16_supported):
    from trl import GRPOConfig

    return GRPOConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        bf16=bfloat16_supported,
        fp16=not bfloat16_supported,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        temperature=args.temperature,
        top_p=args.top_p,
        beta=args.beta,
        epsilon=args.epsilon,
        num_iterations=args.num_iterations,
        reward_weights=REWARD_WEIGHTS,
        scale_rewards=False,
        loss_type="dr_grpo",
        mask_truncated_completions=True,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        seed=args.seed,
        data_seed=args.seed,
        report_to=args.report_to,
        log_completions=args.log_completions,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        use_vllm=False,
    )


def patch_qwen35_generation_inputs(model):
    """Expose mm_token_type_ids to Transformers generation validation.

    Unsloth 2026.6.9 replaces the Qwen3.5 forward method with a compiled wrapper
    whose introspected signature omits this valid multimodal argument.
    """
    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    if getattr(base_model.config, "model_type", None) != "qwen3_5":
        return False
    original_prepare = base_model.prepare_inputs_for_generation
    if "mm_token_type_ids" in inspect.signature(original_prepare).parameters:
        return False

    def prepare_inputs_for_generation(
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        is_first_iteration=False,
        mm_token_type_ids=None,
        **kwargs,
    ):
        if position_ids is None and mm_token_type_ids is not None and (
            image_grid_thw is not None or video_grid_thw is not None
        ):
            position_ids = base_model._prepare_position_ids_for_generation(
                input_ids,
                {
                    "attention_mask": attention_mask,
                    "mm_token_type_ids": mm_token_type_ids,
                    "image_grid_thw": image_grid_thw,
                    "video_grid_thw": video_grid_thw,
                },
            )
        return original_prepare(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            use_cache=use_cache,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            is_first_iteration=is_first_iteration,
            mm_token_type_ids=mm_token_type_ids,
            **kwargs,
        )

    base_model.prepare_inputs_for_generation = prepare_inputs_for_generation
    return True


def patch_qwen35_training_linear_attention(model):
    """Avoid an FLA backward kernel that exceeds H200 shared memory."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        torch_chunk_gated_delta_rule,
    )

    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    patched = 0
    for module in base_model.modules():
        if type(module).__name__ != "Qwen3_5GatedDeltaNet":
            continue
        module.chunk_gated_delta_rule = torch_chunk_gated_delta_rule
        patched += 1
    return patched


def main(argv=None):
    args = parse_args(argv)
    validate_args(args)
    prompt = load_prompt(args.prompt_file)
    train_dataset, eval_dataset = load_datasets(args, prompt)
    print_summary(train_dataset, eval_dataset, args)
    if args.dry_run:
        print("Dry run passed: dataset, prompt, image count, and reference HTML are valid.")
        return

    os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
    # Unsloth must patch Torch, Transformers, and TRL before they are imported.
    import unsloth  # noqa: F401
    from trl import GRPOTrainer
    from unsloth import FastVisionModel, is_bfloat16_supported

    model, processor = FastVisionModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=not args.load_in_16bit,
        load_in_16bit=args.load_in_16bit,
        use_gradient_checkpointing="unsloth",
        max_lora_rank=max_lora_rank_for_model(args.model_name, args.lora_r),
    )
    if hasattr(model, "peft_config"):
        print(f"Continuing the existing LoRA adapter from {args.model_name}.")
    else:
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
    patch_qwen35_generation_inputs(model)
    patch_qwen35_training_linear_attention(model)
    FastVisionModel.for_training(model)

    trainer = GRPOTrainer(
        model=model,
        processing_class=processor,
        reward_funcs=REWARD_FUNCTIONS,
        args=make_config(args, is_bfloat16_supported()),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
