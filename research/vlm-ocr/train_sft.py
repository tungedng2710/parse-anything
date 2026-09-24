#!/usr/bin/env python3
"""Fine-tune aligned Chandra with a trainable merger and language-model LoRA."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

# Unsloth must patch Transformers before Transformers or PEFT is imported.
try:
    from unsloth import FastVisionModel as UnslothFastVisionModel

    UNSLOTH_IMPORT_ERROR: Exception | None = None
except Exception as error:  # pragma: no cover - depends on CUDA/platform support
    UnslothFastVisionModel = None
    UNSLOTH_IMPORT_ERROR = error

import torch
import yaml
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
from safetensors.torch import save_file
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    get_cosine_schedule_with_warmup,
)

from chandra_mae.checkpoint import apply_vision_delta
from chandra_alignment.tracking import (
    init_trackers,
    project_configuration,
    tracker_names,
)
from train_alignment import (
    evaluate,
    get_visual,
    make_loader,
    move_batch,
    rotate_checkpoints,
    set_seed,
    unpack_collated_batch,
)

DEFAULT_LORA_TARGETS = ",".join(
    (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "in_proj_qkv",
        "in_proj_z",
        "in_proj_b",
        "in_proj_a",
        "out_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
)


@dataclass
class SFTConfig:
    dataset: str = "alignment_dataset"
    images_subdir: str = "images"
    labels_subdir: str = "labels"
    prompt: str = "Extract the document as JSON. Return JSON only."
    validation_fraction: float = 0.02
    document_id_regex: str = r"_page\d+$"
    base_model: str | None = None
    load_adapted_vision: bool = True
    mae_delta: str | None = None
    alignment_delta: str | None = None
    alignment_dir: str = "outputs/chandra2-alignment"
    output_dir: str = "outputs/chandra2-sft"
    backend: str = "unsloth"
    unsloth_fullgraph: bool = True
    min_pixels: int = 65_536
    max_pixels: int = 589_824
    max_sequence_length: int = 8_192
    overlength_policy: str = "skip"
    attention_implementation: str = "sdpa"
    batch_size: int = 1
    gradient_accumulation_steps: int = 32
    max_steps: int = 2_000
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_target_modules: str = DEFAULT_LORA_TARGETS
    lora_learning_rate: float = 1e-4
    merger_learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    mixed_precision: str = "bf16"
    gradient_checkpointing: bool = True
    shuffle_buffer: int = 2_000
    num_workers: int = 2
    log_every: int = 1
    eval_every: int = 250
    eval_batches: int = 25
    save_every: int = 250
    keep_last_checkpoints: int = 3
    seed: int = 42
    local_files_only: bool = False
    resume_from: str | None = None
    report_to: str = "tensorboard"
    tracker_project_name: str = "chandra-sft"
    tracker_run_name: str | None = None


def parse_args() -> SFTConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="YAML SFT configuration")
    for field in fields(SFTConfig):
        option = "--" + field.name.replace("_", "-")
        if field.type is bool or isinstance(field.default, bool):
            parser.add_argument(
                option, action=argparse.BooleanOptionalAction, default=None
            )
        else:
            value_type = type(field.default) if field.default is not None else str
            parser.add_argument(option, type=value_type, default=None)
    namespace = vars(parser.parse_args())
    config_path = namespace.pop("config")
    values: dict[str, Any] = {}
    if config_path:
        loaded = yaml.safe_load(config_path.read_text()) or {}
        unknown = set(loaded) - {field.name for field in fields(SFTConfig)}
        if unknown:
            parser.error(f"Unknown configuration keys: {', '.join(sorted(unknown))}")
        values.update(loaded)
    values.update({key: value for key, value in namespace.items() if value is not None})
    config = SFTConfig(**values)
    positive = (
        "batch_size",
        "gradient_accumulation_steps",
        "max_steps",
        "max_sequence_length",
        "shuffle_buffer",
        "lora_rank",
        "lora_alpha",
        "log_every",
        "eval_every",
        "eval_batches",
        "save_every",
    )
    invalid = [name for name in positive if getattr(config, name) <= 0]
    if invalid:
        parser.error(f"These values must be positive: {', '.join(invalid)}")
    if config.num_workers < 0:
        parser.error("num_workers cannot be negative")
    if not 0.0 < config.validation_fraction < 1.0:
        parser.error("validation_fraction must be between 0 and 1")
    if config.overlength_policy not in {"skip", "error"}:
        parser.error("overlength_policy must be one of: skip, error")
    if config.mixed_precision not in {"no", "fp16", "bf16"}:
        parser.error("mixed_precision must be one of: no, fp16, bf16")
    if config.backend not in {"unsloth", "transformers"}:
        parser.error("backend must be one of: unsloth, transformers")
    if not 0.0 <= config.lora_dropout < 1.0:
        parser.error("lora_dropout must be in [0, 1)")
    if not [item for item in config.lora_target_modules.split(",") if item.strip()]:
        parser.error("lora_target_modules cannot be empty")
    try:
        tracker_names(config.report_to)
    except ValueError as error:
        parser.error(str(error))
    return config


def load_model_and_processor(
    config: SFTConfig, base_model: str, dtype: torch.dtype
) -> tuple[torch.nn.Module, Any]:
    if config.backend == "unsloth":
        if UnslothFastVisionModel is None:
            raise RuntimeError(
                "Unsloth could not be imported. Install the Unsloth extra or use "
                "--backend transformers. Original import error: "
                f"{UNSLOTH_IMPORT_ERROR}"
            )
        model, processor = UnslothFastVisionModel.from_pretrained(
            model_name=base_model,
            max_seq_length=config.max_sequence_length,
            dtype=dtype,
            load_in_4bit=False,
            load_in_16bit=True,
            use_gradient_checkpointing=(
                "unsloth" if config.gradient_checkpointing else False
            ),
            fullgraph=config.unsloth_fullgraph,
            local_files_only=config.local_files_only,
        )
        if not hasattr(processor, "apply_chat_template"):
            raise RuntimeError("Unsloth did not return a multimodal Chandra processor")
        return model, processor

    processor = AutoProcessor.from_pretrained(
        base_model, local_files_only=config.local_files_only
    )
    model = AutoModelForImageTextToText.from_pretrained(
        base_model,
        dtype=dtype,
        attn_implementation=config.attention_implementation,
        local_files_only=config.local_files_only,
    )
    return model, processor


def add_lora_adapter(
    model: torch.nn.Module, config: SFTConfig, target_modules: list[str]
) -> torch.nn.Module:
    if config.backend == "unsloth":
        if UnslothFastVisionModel is None:  # Defensive; loader reports the detail.
            raise RuntimeError("Unsloth is unavailable")
        return UnslothFastVisionModel.get_peft_model(
            model,
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=target_modules,
            bias="none",
            finetune_vision_layers=False,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            use_gradient_checkpointing=(
                "unsloth" if config.gradient_checkpointing else False
            ),
            random_state=config.seed,
        )
    return get_peft_model(
        model,
        LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.is_file() else {}


def resolve_sft_inputs(
    config: SFTConfig,
) -> tuple[str, Path | None, Path | None, dict[str, Any]]:
    alignment_dir = Path(config.alignment_dir).expanduser()
    alignment_delta = (
        Path(config.alignment_delta).expanduser()
        if config.alignment_delta
        else alignment_dir / "alignment_merger_delta.safetensors"
    )
    manifest = _read_json(alignment_dir / "alignment_delta_manifest.json")
    alignment_config = _read_json(alignment_dir / "alignment_config.json")
    base_model = (
        config.base_model
        or manifest.get("base_model")
        or alignment_config.get("base_model")
    )
    if not base_model:
        raise ValueError("Could not resolve the base model; pass --base-model")
    if not config.load_adapted_vision:
        return base_model, None, None, manifest

    mae_value = (
        config.mae_delta
        or manifest.get("mae_delta")
        or alignment_config.get("mae_delta")
    )
    if not mae_value:
        raise ValueError("Could not resolve the MAE delta; pass --mae-delta")
    mae_delta = Path(mae_value).expanduser()
    for label, path in (("MAE delta", mae_delta), ("alignment delta", alignment_delta)):
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    manifest_model = manifest.get("base_model")
    if manifest_model and manifest_model != base_model:
        raise ValueError(
            f"Alignment delta expects {manifest_model!r}, received {base_model!r}"
        )
    return base_model, mae_delta, alignment_delta, manifest


def configure_sft_parameters(
    model: torch.nn.Module,
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    lora_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if "lora_" in name and parameter.requires_grad
    ]
    base_model = model.get_base_model()
    visual = get_visual(base_model)
    visual.requires_grad_(False)
    visual.merger.requires_grad_(True)
    merger_parameters = list(visual.merger.parameters())
    expected = {id(parameter) for parameter in lora_parameters + merger_parameters}
    unexpected = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and id(parameter) not in expected
    ]
    if unexpected:
        raise RuntimeError(f"Unexpected SFT trainable parameters: {unexpected[:5]}")
    if not lora_parameters:
        raise RuntimeError("No LoRA parameters were created")
    return lora_parameters, merger_parameters


def save_sft_artifacts(
    model: torch.nn.Module,
    processor: Any,
    output_dir: Path,
    config: SFTConfig,
    base_model: str,
    mae_delta: Path | None,
    alignment_delta: Path | None,
    completed_steps: int,
) -> None:
    adapter_dir = output_dir / "lora_adapter"
    model.save_pretrained(adapter_dir, safe_serialization=True)
    processor.save_pretrained(output_dir / "processor")
    visual = get_visual(model.get_base_model())
    merger_path = output_dir / "sft_merger_delta.safetensors"
    save_file(
        {
            f"model.visual.merger.{name}": tensor.detach().cpu().contiguous()
            for name, tensor in visual.merger.state_dict().items()
        },
        merger_path,
        metadata={
            "base_model": base_model,
            "mae_delta": str(mae_delta or ""),
            "alignment_delta": str(alignment_delta or ""),
            "steps": str(completed_steps),
        },
    )
    manifest = {
        "format": "chandra-document-sft-v1",
        "base_model": base_model,
        "backend": config.backend,
        "unsloth_fullgraph": config.unsloth_fullgraph,
        "load_adapted_vision": config.load_adapted_vision,
        "mae_delta": str(mae_delta) if mae_delta else None,
        "alignment_delta": str(alignment_delta) if alignment_delta else None,
        "merger_delta": merger_path.name,
        "lora_adapter": adapter_dir.name,
        "steps": completed_steps,
        "lora_rank": config.lora_rank,
        "lora_target_modules": [
            item.strip()
            for item in config.lora_target_modules.split(",")
            if item.strip()
        ],
    }
    (output_dir / "sft_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    config = parse_args()
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        step_scheduler_with_optimizer=False,
        log_with=tracker_names(config.report_to),
        project_config=project_configuration(config.output_dir),
    )
    set_seed(config.seed + accelerator.process_index)
    base_model, mae_delta, alignment_delta, _ = resolve_sft_inputs(config)
    config.base_model = base_model
    config.mae_delta = str(mae_delta) if mae_delta else None
    config.alignment_delta = str(alignment_delta) if alignment_delta else None
    output_dir = Path(config.output_dir)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "sft_config.json").write_text(
            json.dumps(asdict(config), indent=2) + "\n"
        )
    accelerator.wait_for_everyone()
    init_trackers(accelerator, config, output_dir)

    accelerator.print(f"Base model: {base_model}")
    accelerator.print(
        "Initialization: "
        + (
            f"MAE={mae_delta}, alignment={alignment_delta}"
            if config.load_adapted_vision
            else "original Chandra (direct-SFT baseline)"
        )
    )
    dtype = {
        "no": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[config.mixed_precision]
    model, processor = load_model_and_processor(config, base_model, dtype)
    if mae_delta is not None:
        apply_vision_delta(model, mae_delta)
    if alignment_delta is not None:
        apply_vision_delta(model, alignment_delta)

    target_modules = [
        item.strip() for item in config.lora_target_modules.split(",") if item.strip()
    ]
    model = add_lora_adapter(model, config, target_modules)
    lora_parameters, merger_parameters = configure_sft_parameters(model)
    model.config.use_cache = False
    if config.backend == "unsloth":
        UnslothFastVisionModel.for_training(
            model, use_gradient_checkpointing=config.gradient_checkpointing
        )
    elif config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    optimizer = AdamW(
        [
            {"params": lora_parameters, "lr": config.lora_learning_rate},
            {"params": merger_parameters, "lr": config.merger_learning_rate},
        ],
        weight_decay=config.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=math.ceil(config.max_steps * config.warmup_ratio),
        num_training_steps=config.max_steps,
    )
    train_loader = make_loader(config, processor, accelerator, "train")
    validation_loader = make_loader(config, processor, accelerator, "validation")
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    completed_steps = 0
    if config.resume_from:
        accelerator.load_state(config.resume_from)
        checkpoint_name = Path(config.resume_from).name
        if checkpoint_name.startswith("step-"):
            completed_steps = int(checkpoint_name.removeprefix("step-"))

    effective_batch = (
        config.batch_size
        * accelerator.num_processes
        * config.gradient_accumulation_steps
    )
    trainable_count = sum(
        parameter.numel()
        for parameter in accelerator.unwrap_model(model).parameters()
        if parameter.requires_grad
    )
    accelerator.print(
        f"SFT: trainable={trainable_count:,}, LoRA={sum(p.numel() for p in lora_parameters):,}, "
        f"merger={sum(p.numel() for p in merger_parameters):,}, "
        f"effective_batch={effective_batch:,}, steps={config.max_steps:,}"
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(
        total=config.max_steps,
        initial=completed_steps,
        desc="SFT steps",
        unit="step",
        dynamic_ncols=True,
        disable=not accelerator.is_main_process,
    )
    if accelerator.is_main_process:
        progress.set_postfix_str("waiting for first paired batch")
    window_loss = torch.zeros((), device=accelerator.device)
    window_microbatches = 0
    step_loss = torch.zeros((), device=accelerator.device)
    step_microbatches = 0
    accumulation_step = 0
    skipped_samples = 0
    started = time.monotonic()

    while completed_steps < config.max_steps:
        batches_this_pass = 0
        usable_batches_this_pass = 0
        for raw_batch in train_loader:
            batches_this_pass += 1
            batch, overlength_count, original_count = unpack_collated_batch(raw_batch)
            if accelerator.num_processes > 1:
                counts = accelerator.reduce(
                    torch.tensor(
                        [overlength_count, original_count],
                        device=accelerator.device,
                        dtype=torch.long,
                    ),
                    reduction="sum",
                )
                if int(counts[0].item()):
                    skipped_samples += int(counts[1].item())
                    if accelerator.is_main_process:
                        progress.set_postfix_str(
                            f"skipped={skipped_samples} (overlength)"
                        )
                    continue
            else:
                skipped_samples += (
                    overlength_count if batch is not None else original_count
                )
            if batch is None:
                if accelerator.is_main_process:
                    progress.set_postfix_str(f"skipped={skipped_samples} (overlength)")
                continue

            usable_batches_this_pass += 1
            batch = move_batch(batch, accelerator.device)
            with accelerator.accumulate(model):
                output = model(**batch)
                accelerator.backward(output.loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model.parameters(), config.max_grad_norm
                    )
                optimizer.step()
                if accelerator.sync_gradients:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            detached_loss = output.loss.detach().float()
            window_loss += detached_loss
            window_microbatches += 1
            step_loss += detached_loss
            step_microbatches += 1
            accumulation_step += 1
            if not accelerator.sync_gradients:
                if accelerator.is_main_process:
                    progress.set_postfix_str(
                        f"accumulation={accumulation_step}/"
                        f"{config.gradient_accumulation_steps}, "
                        f"skipped={skipped_samples}"
                    )
                continue

            completed_steps += 1
            reduced_step_loss = accelerator.reduce(
                step_loss / max(1, step_microbatches), reduction="mean"
            ).item()
            step_loss.zero_()
            step_microbatches = 0
            accumulation_step = 0
            if accelerator.is_main_process:
                progress.update(1)
                progress.set_postfix(
                    loss=f"{reduced_step_loss:.4f}",
                    lora_lr=f"{scheduler.get_last_lr()[0]:.2e}",
                    merger_lr=f"{scheduler.get_last_lr()[1]:.2e}",
                    skipped=skipped_samples,
                )

            if completed_steps % config.log_every == 0:
                train_loss = accelerator.reduce(
                    window_loss / max(1, window_microbatches), reduction="mean"
                ).item()
                metrics = {
                    "step": completed_steps,
                    "train_loss": train_loss,
                    "lora_lr": scheduler.get_last_lr()[0],
                    "merger_lr": scheduler.get_last_lr()[1],
                    "skipped_overlength_samples": skipped_samples,
                    "elapsed_seconds": time.monotonic() - started,
                }
                if accelerator.is_main_process:
                    with (output_dir / "metrics.jsonl").open("a") as handle:
                        handle.write(json.dumps(metrics) + "\n")
                    progress.write(json.dumps(metrics))
                accelerator.log(metrics, step=completed_steps)
                window_loss.zero_()
                window_microbatches = 0

            if completed_steps % config.eval_every == 0:
                validation_loss, validation_skipped = evaluate(
                    model, validation_loader, accelerator, config.eval_batches
                )
                if accelerator.is_main_process:
                    metrics = {
                        "step": completed_steps,
                        "validation_loss": validation_loss,
                        "validation_skipped_overlength_samples": validation_skipped,
                    }
                    with (output_dir / "metrics.jsonl").open("a") as handle:
                        handle.write(json.dumps(metrics) + "\n")
                    progress.write(json.dumps(metrics))
                accelerator.log(metrics, step=completed_steps)

            if completed_steps % config.save_every == 0:
                checkpoint_dir = output_dir / "checkpoints" / f"step-{completed_steps}"
                accelerator.save_state(checkpoint_dir)
                if accelerator.is_main_process:
                    rotate_checkpoints(
                        output_dir / "checkpoints", config.keep_last_checkpoints
                    )
            if completed_steps >= config.max_steps:
                break

        if batches_this_pass == 0:
            raise RuntimeError(
                "The SFT loader produced no batches. Check pairing, split, batch size, "
                "and distributed sharding."
            )
        if usable_batches_this_pass == 0:
            raise RuntimeError(
                "Every SFT batch was overlength. Increase max_sequence_length or "
                "remove/split the overlength labels."
            )

    progress.close()
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_sft_artifacts(
            accelerator.unwrap_model(model),
            processor,
            output_dir,
            config,
            base_model,
            mae_delta,
            alignment_delta,
            completed_steps,
        )
        accelerator.print(f"Saved SFT artifacts to {output_dir}")
        accelerator.print(
            f"Skipped {skipped_samples:,} samples due to overlength filtering"
        )
    accelerator.end_training()


if __name__ == "__main__":
    main()
