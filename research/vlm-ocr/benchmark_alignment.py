#!/usr/bin/env python3
"""Benchmark base, MAE, and aligned Chandra variants on paired JSON data."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import yaml
from tqdm.auto import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

from chandra_alignment import PairedJsonDataset
from chandra_alignment.metrics import aggregate_scores, score_json_prediction
from chandra_mae.checkpoint import apply_vision_delta


@dataclass
class BenchmarkConfig:
    dataset: str | None = None
    images_subdir: str = "images"
    labels_subdir: str = "labels"
    prompt: str = "Extract the document as JSON. Return JSON only."
    validation_fraction: float = 0.02
    document_id_regex: str = r"_page\d+$"
    split: str = "validation"
    base_model: str | None = None
    mae_delta: str | None = None
    alignment_delta: str | None = None
    alignment_dir: str = "outputs/chandra2-alignment"
    output_dir: str = "outputs/chandra2-alignment-benchmark"
    variants: str = "base,mae,aligned"
    max_samples: int = 200
    batch_size: int = 1
    min_pixels: int = 65_536
    max_pixels: int = 589_824
    max_input_length: int = 8_192
    max_new_tokens: int = 8_192
    attention_implementation: str = "sdpa"
    mixed_precision: str = "bf16"
    local_files_only: bool = False
    seed: int = 42


def parse_args() -> BenchmarkConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="YAML benchmark configuration")
    for field in fields(BenchmarkConfig):
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
        unknown = set(loaded) - {field.name for field in fields(BenchmarkConfig)}
        if unknown:
            parser.error(f"Unknown configuration keys: {', '.join(sorted(unknown))}")
        values.update(loaded)
    values.update({key: value for key, value in namespace.items() if value is not None})
    config = BenchmarkConfig(**values)
    if config.split not in {"train", "validation"}:
        parser.error("split must be train or validation")
    if config.max_samples <= 0 or config.batch_size <= 0:
        parser.error("max_samples and batch_size must be positive")
    if config.max_input_length <= 0 or config.max_new_tokens <= 0:
        parser.error("token limits must be positive")
    if config.mixed_precision not in {"fp16", "bf16", "no"}:
        parser.error("mixed_precision must be one of: no, fp16, bf16")
    requested = [value.strip() for value in config.variants.split(",") if value.strip()]
    if not requested or set(requested) - {"base", "mae", "aligned"}:
        parser.error("variants must be a comma-separated subset of base,mae,aligned")
    return config


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.is_file() else {}


def resolve_artifacts(config: BenchmarkConfig) -> tuple[str, Path, Path]:
    alignment_dir = Path(config.alignment_dir).expanduser()
    alignment_delta = (
        Path(config.alignment_delta).expanduser()
        if config.alignment_delta
        else alignment_dir / "alignment_merger_delta.safetensors"
    )
    alignment_manifest = _read_json(
        alignment_delta.with_name("alignment_delta_manifest.json")
    )
    alignment_config = _read_json(alignment_dir / "alignment_config.json")
    base_model = (
        config.base_model
        or alignment_manifest.get("base_model")
        or alignment_config.get("base_model")
    )
    mae_value = (
        config.mae_delta
        or alignment_manifest.get("mae_delta")
        or alignment_config.get("mae_delta")
    )
    if config.dataset is None:
        config.dataset = alignment_config.get("dataset")
    if not base_model or not mae_value:
        raise ValueError(
            "Could not resolve base_model or mae_delta from alignment output"
        )
    mae_delta = Path(mae_value).expanduser()
    for label, path in (("MAE delta", mae_delta), ("alignment delta", alignment_delta)):
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    if not config.dataset:
        raise ValueError(
            "Dataset was not provided and is absent from alignment_config.json"
        )
    return base_model, mae_delta, alignment_delta


def make_dataset(config: BenchmarkConfig) -> PairedJsonDataset:
    return PairedJsonDataset(
        root=config.dataset,
        split=config.split,
        images_subdir=config.images_subdir,
        labels_subdir=config.labels_subdir,
        validation_fraction=config.validation_fraction,
        document_id_regex=config.document_id_regex,
        seed=config.seed,
        shuffle_buffer=1,
    )


def batches(
    iterator: Iterator[dict[str, Any]], batch_size: int, limit: int
) -> Iterator[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    seen = 0
    for example in iterator:
        if seen >= limit:
            break
        batch.append(example)
        seen += 1
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def generation_prompts(processor: Any, prompt: str, count: int) -> list[str]:
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    return [text] * count


def evaluate_variant(
    name: str,
    model: torch.nn.Module,
    processor: Any,
    config: BenchmarkConfig,
    device: torch.device,
) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    prediction_path = output_dir / f"{name}.jsonl"
    scores: list[dict[str, Any]] = []
    completed = 0
    autocast_enabled = device.type == "cuda" and config.mixed_precision != "no"
    autocast_dtype = (
        torch.bfloat16 if config.mixed_precision == "bf16" else torch.float16
    )
    with (
        prediction_path.open("w", encoding="utf-8") as output,
        tqdm(
            total=config.max_samples, desc=name, unit="sample", dynamic_ncols=True
        ) as progress,
    ):
        for examples in batches(
            iter(make_dataset(config)), config.batch_size, config.max_samples
        ):
            inputs = processor(
                images=[example["image"] for example in examples],
                text=generation_prompts(processor, config.prompt, len(examples)),
                padding=True,
                truncation=True,
                max_length=config.max_input_length,
                min_pixels=config.min_pixels,
                max_pixels=config.max_pixels,
                return_tensors="pt",
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with (
                torch.inference_mode(),
                torch.autocast(
                    device_type=device.type,
                    dtype=autocast_dtype,
                    enabled=autocast_enabled,
                ),
            ):
                generated = model.generate(
                    **inputs,
                    max_new_tokens=config.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )
            prompt_length = inputs["input_ids"].shape[1]
            predictions = processor.batch_decode(
                generated[:, prompt_length:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for example, prediction in zip(examples, predictions, strict=True):
                reference = json.loads(example["target"])
                score = score_json_prediction(prediction, reference)
                scores.append(score)
                record = {
                    "variant": name,
                    "source": example["source"],
                    "reference": reference,
                    "prediction": prediction,
                    **score,
                }
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
            completed += len(examples)
            progress.update(len(examples))
            summary = aggregate_scores(scores)
            progress.set_postfix(
                valid=f"{summary['json_valid_rate']:.3f}",
                field_f1=f"{summary['field_f1']:.3f}",
            )
    summary = aggregate_scores(scores)
    summary["prediction_file"] = str(prediction_path)
    summary["completed_samples"] = completed
    return summary


def main() -> None:
    config = parse_args()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    base_model, mae_delta, alignment_delta = resolve_artifacts(config)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "benchmark_config.json").write_text(
        json.dumps(asdict(config), indent=2) + "\n"
    )

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for Chandra generation benchmarking")
    device = torch.device("cuda")
    dtype = {
        "no": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[config.mixed_precision]
    processor = AutoProcessor.from_pretrained(
        base_model, local_files_only=config.local_files_only
    )
    processor.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(
        base_model,
        dtype=dtype,
        attn_implementation=config.attention_implementation,
        local_files_only=config.local_files_only,
    ).to(device)
    model.eval()

    requested = {value.strip() for value in config.variants.split(",") if value.strip()}
    summaries: dict[str, Any] = {}
    if "base" in requested:
        summaries["base"] = evaluate_variant("base", model, processor, config, device)
    apply_vision_delta(model, mae_delta)
    if "mae" in requested:
        summaries["mae"] = evaluate_variant("mae", model, processor, config, device)
    apply_vision_delta(model, alignment_delta)
    if "aligned" in requested:
        summaries["aligned"] = evaluate_variant(
            "aligned", model, processor, config, device
        )

    report = {
        "base_model": base_model,
        "mae_delta": str(mae_delta),
        "alignment_delta": str(alignment_delta),
        "results": summaries,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
