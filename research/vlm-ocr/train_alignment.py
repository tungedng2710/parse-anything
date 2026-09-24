#!/usr/bin/env python3
"""Align Chandra's merger to an MAE-adapted vision encoder using image/JSON pairs."""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from accelerate import Accelerator
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    get_cosine_schedule_with_warmup,
)

from chandra_alignment import (
    ORIGINAL_COUNT_KEY,
    OVERLENGTH_COUNT_KEY,
    AlignmentCollator,
    PairedJsonDataset,
)
from chandra_alignment.tracking import (
    init_trackers,
    project_configuration,
    tracker_names,
)
from chandra_mae.checkpoint import apply_vision_delta


@dataclass
class AlignmentConfig:
    dataset: str = "alignment_dataset"
    images_subdir: str = "images"
    labels_subdir: str = "labels"
    prompt: str = "Extract the document as JSON. Return JSON only."
    validation_fraction: float = 0.02
    document_id_regex: str = r"_page\d+$"
    base_model: str | None = None
    mae_delta: str | None = None
    output_dir: str = "outputs/chandra2-alignment"
    min_pixels: int = 65_536
    max_pixels: int = 589_824
    max_sequence_length: int = 8_192
    overlength_policy: str = "skip"
    attention_implementation: str = "sdpa"
    batch_size: int = 2
    gradient_accumulation_steps: int = 32
    max_steps: int = 2_000
    merger_learning_rate: float = 1e-4
    train_vision: bool = False
    vision_learning_rate: float = 2e-6
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
    tracker_project_name: str = "chandra-alignment"
    tracker_run_name: str | None = None


def parse_args() -> AlignmentConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="YAML configuration file")
    for field in fields(AlignmentConfig):
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
        unknown = set(loaded) - {field.name for field in fields(AlignmentConfig)}
        if unknown:
            parser.error(f"Unknown configuration keys: {', '.join(sorted(unknown))}")
        values.update(loaded)
    values.update({key: value for key, value in namespace.items() if value is not None})
    config = AlignmentConfig(**values)
    positive = (
        "batch_size",
        "gradient_accumulation_steps",
        "max_steps",
        "max_sequence_length",
        "shuffle_buffer",
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
    if config.mixed_precision not in {"no", "fp16", "bf16"}:
        parser.error("mixed_precision must be one of: no, fp16, bf16")
    if config.overlength_policy not in {"skip", "error"}:
        parser.error("overlength_policy must be one of: skip, error")
    try:
        tracker_names(config.report_to)
    except ValueError as error:
        parser.error(str(error))
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def discover_mae_delta(outputs_root: str | Path = "outputs") -> Path:
    candidates = sorted(
        Path(outputs_root).glob("**/chandra_vision_delta.safetensors"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            "No chandra_vision_delta.safetensors found under ./outputs; pass --mae-delta"
        )
    return candidates[0]


def read_delta_manifest(delta: Path) -> dict[str, Any]:
    manifest_path = delta.with_name("vision_delta_manifest.json")
    if not manifest_path.is_file():
        return {}
    return json.loads(manifest_path.read_text())


def resolve_model_inputs(config: AlignmentConfig) -> tuple[str, Path, dict[str, Any]]:
    delta = (
        Path(config.mae_delta).expanduser()
        if config.mae_delta
        else discover_mae_delta()
    )
    if not delta.is_file():
        raise FileNotFoundError(f"MAE delta does not exist: {delta}")
    manifest = read_delta_manifest(delta)
    base_model = config.base_model or manifest.get("base_model")
    if not base_model:
        raise ValueError("Could not determine the base model; pass --base-model")
    manifest_model = manifest.get("base_model")
    if manifest_model and manifest_model != base_model:
        raise ValueError(
            f"MAE delta expects base model {manifest_model!r}, received {base_model!r}"
        )
    return base_model, delta, manifest


def get_visual(model: torch.nn.Module) -> torch.nn.Module:
    base = getattr(model, "model", None)
    visual = getattr(base, "visual", None)
    if visual is None or not hasattr(visual, "merger"):
        raise RuntimeError("Could not locate model.model.visual.merger in Chandra")
    return visual


def configure_trainable_parameters(
    model: torch.nn.Module, train_vision: bool
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    model.requires_grad_(False)
    visual = get_visual(model)
    visual.merger.requires_grad_(True)
    merger_parameters = list(visual.merger.parameters())
    vision_parameters: list[torch.nn.Parameter] = []
    if train_vision:
        visual.requires_grad_(True)
        merger_ids = {id(parameter) for parameter in merger_parameters}
        vision_parameters = [
            parameter
            for parameter in visual.parameters()
            if id(parameter) not in merger_ids
        ]
    trainable_ids = {
        id(parameter) for parameter in merger_parameters + vision_parameters
    }
    unexpected = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and id(parameter) not in trainable_ids
    ]
    if unexpected:
        raise RuntimeError(f"Unexpected trainable parameters: {unexpected[:5]}")
    return merger_parameters, vision_parameters


def make_loader(
    config: AlignmentConfig,
    processor: Any,
    accelerator: Accelerator,
    split: str,
) -> DataLoader:
    dataset = PairedJsonDataset(
        root=config.dataset,
        split=split,
        images_subdir=config.images_subdir,
        labels_subdir=config.labels_subdir,
        validation_fraction=config.validation_fraction,
        document_id_regex=config.document_id_regex,
        seed=config.seed,
        shuffle_buffer=config.shuffle_buffer if split == "train" else 1,
        process_index=accelerator.process_index,
        num_processes=accelerator.num_processes,
    )
    collator = AlignmentCollator(
        processor=processor,
        prompt=config.prompt,
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
        max_sequence_length=config.max_sequence_length,
        overlength_policy=config.overlength_policy,
    )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
        collate_fn=collator,
        drop_last=split == "train",
    )


def move_batch(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def unpack_collated_batch(
    batch: dict[str, Any],
) -> tuple[dict[str, torch.Tensor] | None, int, int]:
    """Remove collator metadata and report overlength/original sample counts."""
    overlength_count = int(batch.pop(OVERLENGTH_COUNT_KEY).item())
    original_count = int(batch.pop(ORIGINAL_COUNT_KEY).item())
    return (batch or None), overlength_count, original_count


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    accelerator: Accelerator,
    max_batches: int,
) -> tuple[float, int]:
    evaluation_model = accelerator.unwrap_model(model)
    evaluation_model.eval()
    loss_sum = torch.zeros((), device=accelerator.device, dtype=torch.float64)
    example_count = torch.zeros((), device=accelerator.device, dtype=torch.float64)
    skipped_count = torch.zeros((), device=accelerator.device, dtype=torch.float64)
    for index, raw_batch in enumerate(loader):
        if index >= max_batches:
            break
        batch, overlength_count, original_count = unpack_collated_batch(raw_batch)
        skipped_count += overlength_count if batch is not None else original_count
        if batch is None:
            continue
        batch = move_batch(batch, accelerator.device)
        output = evaluation_model(**batch)
        count = batch["input_ids"].shape[0]
        loss_sum += output.loss.detach().double() * count
        example_count += count
    totals = torch.stack((loss_sum, example_count, skipped_count))
    totals = accelerator.reduce(totals, reduction="sum")
    evaluation_model.train()
    if totals[1].item() == 0:
        return float("nan"), int(totals[2].item())
    return (totals[0] / totals[1]).item(), int(totals[2].item())


def rotate_checkpoints(root: Path, keep: int) -> None:
    checkpoints = sorted(
        root.glob("step-*"), key=lambda path: int(path.name.removeprefix("step-"))
    )
    for checkpoint in checkpoints[:-keep] if keep > 0 else checkpoints:
        shutil.rmtree(checkpoint)


def save_alignment_delta(
    model: torch.nn.Module,
    output_dir: Path,
    base_model: str,
    mae_delta: Path,
    steps: int,
    train_vision: bool,
) -> Path:
    visual = get_visual(model)
    if train_vision:
        filename = "aligned_vision_delta.safetensors"
        tensors = {
            f"model.visual.{name}": tensor.detach().cpu().contiguous()
            for name, tensor in visual.state_dict().items()
        }
    else:
        filename = "alignment_merger_delta.safetensors"
        tensors = {
            f"model.visual.merger.{name}": tensor.detach().cpu().contiguous()
            for name, tensor in visual.merger.state_dict().items()
        }
    path = output_dir / filename
    save_file(
        tensors,
        path,
        metadata={
            "base_model": base_model,
            "mae_delta": str(mae_delta),
            "steps": str(steps),
            "train_vision": str(train_vision).lower(),
        },
    )
    manifest = {
        "format": "chandra-alignment-delta-v1",
        "base_model": base_model,
        "mae_delta": str(mae_delta),
        "steps": steps,
        "train_vision": train_vision,
        "tensor_count": len(tensors),
        "delta": filename,
    }
    (output_dir / "alignment_delta_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return path


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
    base_model, mae_delta, mae_manifest = resolve_model_inputs(config)
    config.base_model = base_model
    config.mae_delta = str(mae_delta)
    output_dir = Path(config.output_dir)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "alignment_config.json").write_text(
            json.dumps(asdict(config), indent=2) + "\n"
        )
    accelerator.wait_for_everyone()
    init_trackers(accelerator, config, output_dir)

    accelerator.print(f"Base model: {base_model}")
    accelerator.print(f"MAE delta: {mae_delta}")
    if mae_manifest:
        accelerator.print(
            f"MAE source: steps={mae_manifest.get('steps')}, "
            f"mask_ratio={mae_manifest.get('mask_ratio')}"
        )

    processor = AutoProcessor.from_pretrained(
        base_model, local_files_only=config.local_files_only
    )
    dtype = {
        "no": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[config.mixed_precision]
    model = AutoModelForImageTextToText.from_pretrained(
        base_model,
        dtype=dtype,
        attn_implementation=config.attention_implementation,
        local_files_only=config.local_files_only,
    )
    apply_vision_delta(model, mae_delta)
    merger_parameters, vision_parameters = configure_trainable_parameters(
        model, config.train_vision
    )
    model.config.use_cache = False
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    parameter_groups = [
        {"params": merger_parameters, "lr": config.merger_learning_rate}
    ]
    if vision_parameters:
        parameter_groups.append(
            {"params": vision_parameters, "lr": config.vision_learning_rate}
        )
    optimizer = AdamW(
        parameter_groups,
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
        f"Alignment: trainable={trainable_count:,}, effective_batch={effective_batch:,}, "
        f"steps={config.max_steps:,}, train_vision={config.train_vision}"
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(
        total=config.max_steps,
        initial=completed_steps,
        desc="alignment steps",
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
                counts = torch.tensor(
                    [overlength_count, original_count],
                    device=accelerator.device,
                    dtype=torch.long,
                )
                counts = accelerator.reduce(counts, reduction="sum")
                global_overlength = int(counts[0].item())
                global_original = int(counts[1].item())
                if global_overlength:
                    # Filtering different rows on different ranks changes local
                    # batch sizes and gradient weighting. Drop this synchronized
                    # microbatch everywhere instead.
                    skipped_samples += global_original
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
                    merger_lr=f"{scheduler.get_last_lr()[0]:.2e}",
                    skipped=skipped_samples,
                )

            if completed_steps % config.log_every == 0:
                train_loss = accelerator.reduce(
                    window_loss / max(1, window_microbatches), reduction="mean"
                ).item()
                metrics = {
                    "step": completed_steps,
                    "train_loss": train_loss,
                    "merger_lr": scheduler.get_last_lr()[0],
                    "skipped_overlength_samples": skipped_samples,
                    "elapsed_seconds": time.monotonic() - started,
                }
                if vision_parameters:
                    metrics["vision_lr"] = scheduler.get_last_lr()[1]
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
                    evaluation_metrics = {
                        "step": completed_steps,
                        "validation_loss": validation_loss,
                        "validation_skipped_overlength_samples": validation_skipped,
                    }
                    with (output_dir / "metrics.jsonl").open("a") as handle:
                        handle.write(json.dumps(evaluation_metrics) + "\n")
                    progress.write(json.dumps(evaluation_metrics))
                accelerator.log(evaluation_metrics, step=completed_steps)

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
                "The training loader produced no batches. Check image/label pairing, "
                "validation_fraction, batch_size, and distributed sharding."
            )
        if usable_batches_this_pass == 0:
            raise RuntimeError(
                "Every training batch was skipped because its JSON targets exceeded "
                f"max_sequence_length={config.max_sequence_length}. Increase the limit "
                "or remove/split the overlength labels."
            )

    progress.close()
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        delta_path = save_alignment_delta(
            unwrapped,
            output_dir,
            base_model,
            mae_delta,
            completed_steps,
            config.train_vision,
        )
        accelerator.print(f"Saved alignment delta to {delta_path}")
        accelerator.print(f"Skipped {skipped_samples:,} overlength training samples")
    accelerator.end_training()


if __name__ == "__main__":
    main()
