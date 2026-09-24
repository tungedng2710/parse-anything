#!/usr/bin/env python3
"""Train Chandra 2's native vision encoder with DAVE-style raw-pixel MAE."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoProcessor, get_cosine_schedule_with_warmup

from chandra_mae.checkpoint import load_chandra_vision, save_vision_delta
from chandra_mae.data import (
    ChandraImageCollator,
    DocumentImageDataset,
    StreamingDocumentImageDataset,
)
from chandra_mae.model import ChandraMAE


@dataclass
class TrainingConfig:
    model: str = "datalab-to/chandra-ocr-2"
    revision: str | None = None
    dataset: str = "sample_dataset"
    output_dir: str = "outputs/chandra2-mae-pilot"
    seed: int = 42
    recursive: bool = True
    streaming: bool = False
    shuffle_buffer: int = 10_000
    min_pixels: int = 65_536
    max_pixels: int = 589_824
    mask_ratio: float = 0.75
    decoder_hidden_size: int = 384
    decoder_layers: int = 4
    decoder_heads: int = 6
    batch_size: int = 1
    gradient_accumulation_steps: int = 512
    max_steps: int = 5_000
    vision_learning_rate: float = 1e-5
    decoder_learning_rate: float = 1e-4
    weight_decay: float = 0.05
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    mixed_precision: str = "bf16"
    num_workers: int = 4
    log_every: int = 10
    save_every: int = 500
    keep_last_checkpoints: int = 3
    gradient_checkpointing: bool = True
    local_files_only: bool = False
    random_init: bool = False
    resume_from: str | None = None


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="YAML configuration file")
    for field in fields(TrainingConfig):
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
        unknown = set(loaded) - {field.name for field in fields(TrainingConfig)}
        if unknown:
            parser.error(f"Unknown configuration keys: {', '.join(sorted(unknown))}")
        values.update(loaded)
    values.update({key: value for key, value in namespace.items() if value is not None})
    config = TrainingConfig(**values)
    if (
        config.max_steps <= 0
        or config.batch_size <= 0
        or config.gradient_accumulation_steps <= 0
    ):
        parser.error(
            "max_steps, batch_size, and gradient_accumulation_steps must be positive"
        )
    if config.mixed_precision not in {"no", "fp16", "bf16"}:
        parser.error("mixed_precision must be one of: no, fp16, bf16")
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rotate_checkpoints(root: Path, keep: int) -> None:
    checkpoints = sorted(
        (path for path in root.glob("step-*")),
        key=lambda path: int(path.name.split("-")[-1]),
    )
    for checkpoint in checkpoints[:-keep] if keep > 0 else checkpoints:
        import shutil

        shutil.rmtree(checkpoint)


def main() -> None:
    config = parse_args()
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        # This schedule is defined in global optimizer steps. Accelerate's
        # default otherwise advances it once per distributed process.
        step_scheduler_with_optimizer=False,
    )
    set_seed(config.seed + accelerator.process_index)
    output_dir = Path(config.output_dir)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "training_config.json").write_text(
            json.dumps(asdict(config), indent=2) + "\n"
        )

    processor = AutoProcessor.from_pretrained(
        config.model,
        revision=config.revision,
        local_files_only=config.local_files_only,
    )
    vision, base_prefix = load_chandra_vision(
        config.model,
        revision=config.revision,
        local_files_only=config.local_files_only,
        random_init=config.random_init,
    )
    if config.gradient_checkpointing:
        # The vision model inherits the text-model default ("input_ids"), which
        # otherwise makes Transformers look for nonexistent token embeddings.
        vision.main_input_name = "hidden_states"
        vision.gradient_checkpointing_enable()
    model = ChandraMAE(
        vision=vision,
        decoder_hidden_size=config.decoder_hidden_size,
        decoder_layers=config.decoder_layers,
        decoder_heads=config.decoder_heads,
        mask_ratio=config.mask_ratio,
    )
    if config.streaming:
        dataset = StreamingDocumentImageDataset(
            config.dataset,
            recursive=config.recursive,
            seed=config.seed,
            shuffle_buffer=config.shuffle_buffer,
            process_index=accelerator.process_index,
            num_processes=accelerator.num_processes,
        )
    else:
        dataset = DocumentImageDataset(
            config.dataset, recursive=config.recursive, seed=config.seed
        )
    collator = ChandraImageCollator(
        processor.image_processor,
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=not config.streaming,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
        collate_fn=collator,
        drop_last=True,
    )
    vision_parameters = [
        parameter for parameter in model.vision.parameters() if parameter.requires_grad
    ]
    decoder_parameters = list(model.decoder_parameters)
    optimizer = AdamW(
        [
            {"params": vision_parameters, "lr": config.vision_learning_rate},
            {"params": decoder_parameters, "lr": config.decoder_learning_rate},
        ],
        weight_decay=config.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=math.ceil(config.max_steps * config.warmup_ratio),
        num_training_steps=config.max_steps,
    )
    if config.streaming:
        # Packed images have different patch counts, so generic distributed
        # batch slicing cannot preserve pixel_values/grid_thw boundaries. The
        # iterable dataset is rank-sharded above and tensors are moved below.
        model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    else:
        model, optimizer, loader, scheduler = accelerator.prepare(
            model, optimizer, loader, scheduler
        )

    completed_steps = 0
    if config.resume_from:
        accelerator.load_state(config.resume_from)
        name = Path(config.resume_from).name
        if name.startswith("step-"):
            completed_steps = int(name.removeprefix("step-"))

    if accelerator.is_main_process:
        effective_batch = (
            config.batch_size
            * accelerator.num_processes
            * config.gradient_accumulation_steps
        )
        dataset_description = (
            "a streaming corpus" if config.streaming else f"{len(dataset):,} images"
        )
        accelerator.print(
            f"Training on {dataset_description} with {accelerator.num_processes} process(es); "
            f"effective batch={effective_batch:,}, steps={config.max_steps:,}"
        )

    model.train()
    running_loss = 0.0
    micro_steps = 0
    accumulation_step = 0
    started = time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(
        total=config.max_steps,
        initial=completed_steps,
        desc="optimizer steps",
        unit="step",
        dynamic_ncols=True,
        disable=not accelerator.is_main_process,
    )
    if accelerator.is_main_process:
        progress.set_postfix_str("waiting for first batch")
    while completed_steps < config.max_steps:
        for batch in loader:
            if config.streaming:
                batch = {
                    key: value.to(accelerator.device, non_blocking=True)
                    for key, value in batch.items()
                }
            with accelerator.accumulate(model):
                output = model(
                    pixel_values=batch["pixel_values"],
                    grid_thw=batch["grid_thw"],
                    target_patches=batch["target_patches"],
                )
                accelerator.backward(output.loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model.parameters(), config.max_grad_norm
                    )
                optimizer.step()
                if accelerator.sync_gradients:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            running_loss += output.loss.detach().float().item()
            micro_steps += 1
            accumulation_step += 1
            if not accelerator.sync_gradients:
                refresh_every = max(1, config.gradient_accumulation_steps // 100)
                if accelerator.is_main_process and (
                    accumulation_step == 1 or accumulation_step % refresh_every == 0
                ):
                    progress.set_postfix(
                        accumulation=(
                            f"{accumulation_step}/{config.gradient_accumulation_steps}"
                        ),
                        batch_loss=f"{output.loss.detach().float().item():.4f}",
                    )
                continue

            completed_steps += 1
            if accelerator.is_main_process:
                progress.update(1)
                progress.set_postfix(
                    accumulation=(
                        f"{config.gradient_accumulation_steps}/"
                        f"{config.gradient_accumulation_steps}"
                    ),
                    batch_loss=f"{output.loss.detach().float().item():.4f}",
                    vision_lr=f"{scheduler.get_last_lr()[0]:.2e}",
                )
            accumulation_step = 0
            if completed_steps % config.log_every == 0:
                loss = running_loss / max(1, micro_steps)
                loss_tensor = torch.tensor(loss, device=accelerator.device)
                loss = accelerator.reduce(loss_tensor, reduction="mean").item()
                elapsed = time.monotonic() - started
                metrics = {
                    "step": completed_steps,
                    "loss": loss,
                    "vision_lr": scheduler.get_last_lr()[0],
                    "decoder_lr": scheduler.get_last_lr()[1],
                    "elapsed_seconds": elapsed,
                }
                if accelerator.is_main_process:
                    with (output_dir / "metrics.jsonl").open("a") as handle:
                        handle.write(json.dumps(metrics) + "\n")
                    progress.write(json.dumps(metrics))
                running_loss = 0.0
                micro_steps = 0

            if completed_steps % config.save_every == 0:
                checkpoint_dir = output_dir / "checkpoints" / f"step-{completed_steps}"
                accelerator.save_state(checkpoint_dir)
                if accelerator.is_main_process:
                    rotate_checkpoints(
                        output_dir / "checkpoints", config.keep_last_checkpoints
                    )
            if completed_steps >= config.max_steps:
                break

    progress.close()
    accelerator.wait_for_everyone()
    unwrapped = accelerator.unwrap_model(model)
    if accelerator.is_main_process:
        save_vision_delta(
            unwrapped.vision,
            output_dir,
            base_model=config.model,
            base_prefix=base_prefix or "model.visual.",
            metadata={"steps": completed_steps, "mask_ratio": config.mask_ratio},
        )
        torch.save(
            {
                key: value.detach().cpu()
                for key, value in unwrapped.state_dict().items()
                if not key.startswith("vision.")
            },
            output_dir / "mae_decoder.pt",
        )
        accelerator.print(f"Saved adapted vision encoder and delta to {output_dir}")
    accelerator.end_training()


if __name__ == "__main__":
    main()
