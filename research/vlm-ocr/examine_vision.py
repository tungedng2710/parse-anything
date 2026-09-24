#!/usr/bin/env python3
"""Inspect Chandra 2's vision encoder and multimodal projector.

The default mode reads configuration and safetensors metadata only.  Pass
``--run-forward`` to load just the vision tower (not the language model) and
run a synthetic image, or pass ``--image`` to use a real image.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "datalab-to/chandra-ocr-2"
PROJECTOR_NAMES = ("merger", "projector", "mm_projector", "multi_modal_projector")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hub model ID or local directory")
    parser.add_argument("--revision", default=None, help="Optional Hub revision")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not access the network; use an already cached checkpoint",
    )
    parser.add_argument(
        "--config-only",
        action="store_true",
        help="Inspect config without locating/downloading safetensor weights",
    )
    parser.add_argument(
        "--show-tensors",
        type=int,
        default=0,
        metavar="N",
        help="Show the N largest vision tensors",
    )
    parser.add_argument("--run-forward", action="store_true", help="Run the vision tower")
    parser.add_argument("--image", type=Path, help="Image for the forward pass")
    parser.add_argument(
        "--synthetic-size",
        type=int,
        default=224,
        help="Side length of the synthetic image used by --run-forward (default: 224)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Forward-pass device: auto, cpu, cuda, cuda:0, etc. (default: auto)",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = parser.parse_args(argv)
    if args.show_tensors < 0:
        parser.error("--show-tensors must be non-negative")
    if args.synthetic_size <= 0:
        parser.error("--synthetic-size must be positive")
    if args.image is not None:
        args.run_forward = True
    if args.run_forward and args.config_only:
        parser.error("--run-forward requires weights, so it cannot be used with --config-only")
    if args.image is not None and not args.image.is_file():
        parser.error(f"image does not exist: {args.image}")
    return args


def locate_safetensors(model: str, revision: str | None, local_only: bool) -> list[Path]:
    """Resolve all checkpoint shards while avoiding loading their tensor data."""
    local_path = Path(model).expanduser()
    if local_path.is_dir():
        files = sorted(local_path.glob("*.safetensors"))
    else:
        from huggingface_hub import snapshot_download

        snapshot = snapshot_download(
            repo_id=model,
            revision=revision,
            allow_patterns=["*.safetensors", "*.json"],
            local_files_only=local_only,
        )
        files = sorted(Path(snapshot).glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No .safetensors checkpoint found for {model!r}")
    return files


def tensor_metadata(files: list[Path]) -> dict[str, dict[str, Any]]:
    from safetensors import safe_open

    metadata: dict[str, dict[str, Any]] = {}
    for checkpoint in files:
        with safe_open(checkpoint, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                view = handle.get_slice(name)
                shape = list(view.get_shape())
                metadata[name] = {
                    "shape": shape,
                    "dtype": str(view.get_dtype()),
                    "parameters": math.prod(shape),
                    "file": str(checkpoint),
                }
    return metadata


def find_vision_prefix(names: list[str]) -> str | None:
    candidates: Counter[str] = Counter()
    for name in names:
        parts = name.split(".")
        for index, part in enumerate(parts):
            if part in {"visual", "vision_tower", "vision_model", "vision_encoder"}:
                candidates[".".join(parts[: index + 1]) + "."] += 1
    return candidates.most_common(1)[0][0] if candidates else None


def find_projector_prefix(names: list[str], vision_prefix: str | None) -> str | None:
    candidates: Counter[str] = Counter()
    for name in names:
        parts = name.split(".")
        for index, part in enumerate(parts):
            if part in PROJECTOR_NAMES:
                prefix = ".".join(parts[: index + 1]) + "."
                if vision_prefix is None or prefix.startswith(vision_prefix):
                    candidates[prefix] += 1
    return candidates.most_common(1)[0][0] if candidates else None


def parameter_summary(
    metadata: dict[str, dict[str, Any]], prefix: str | None
) -> dict[str, Any] | None:
    if prefix is None:
        return None
    selected = {name: value for name, value in metadata.items() if name.startswith(prefix)}
    return {
        "prefix": prefix.removesuffix("."),
        "parameters": sum(item["parameters"] for item in selected.values()),
        "tensor_count": len(selected),
        "dtypes": dict(Counter(item["dtype"] for item in selected.values())),
    }


def selected_vision_config(config: Any) -> dict[str, Any]:
    vision = config.vision_config
    fields = (
        "depth",
        "hidden_size",
        "intermediate_size",
        "num_heads",
        "patch_size",
        "temporal_patch_size",
        "spatial_merge_size",
        "num_position_embeddings",
        "out_hidden_size",
        "hidden_act",
    )
    return {name: getattr(vision, name) for name in fields if hasattr(vision, name)}


def choose_device(requested: str) -> str:
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return requested


def load_vision_tower(
    config: Any,
    metadata: dict[str, dict[str, Any]],
    files: list[Path],
    vision_prefix: str,
    device: str,
):
    """Materialize only the vision weights, leaving the language model unloaded."""
    import torch
    from accelerate import init_empty_weights
    from safetensors import safe_open
    import transformers

    # Transformers 5.12 currently emits unrelated Qwen3.5 docstring validation
    # errors while lazily importing the model class. Keep inspection output (and
    # especially --json) clean; real import/construction failures still raise.
    previous_verbosity = transformers.logging.get_verbosity()
    transformers.logging.set_verbosity_critical()
    try:
        with init_empty_weights():
            vision = transformers.AutoModel.from_config(config.vision_config)
    finally:
        transformers.logging.set_verbosity(previous_verbosity)

    state = {}
    for checkpoint in files:
        names = [
            name
            for name, info in metadata.items()
            if name.startswith(vision_prefix) and info["file"] == str(checkpoint)
        ]
        if not names:
            continue
        with safe_open(checkpoint, framework="pt", device=device) as handle:
            for name in names:
                state[name.removeprefix(vision_prefix)] = handle.get_tensor(name)

    incompatible = vision.load_state_dict(state, strict=True, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Could not load vision tower exactly: {incompatible}")
    return vision.eval().to(device=device)


def describe_value(value: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        result: dict[str, Any] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype).removeprefix("torch."),
            "device": str(value.device),
        }
        if value.is_floating_point() and value.numel():
            sample = value.detach().float()
            result.update(mean=sample.mean().item(), std=sample.std().item())
        return result
    if isinstance(value, (tuple, list)):
        return [describe_value(item) for item in value]
    return type(value).__name__


def run_forward(
    args: argparse.Namespace,
    config: Any,
    metadata: dict[str, dict[str, Any]],
    files: list[Path],
    vision_prefix: str,
) -> dict[str, Any]:
    import torch
    from PIL import Image
    from transformers import AutoProcessor

    device = choose_device(args.device)
    vision = load_vision_tower(config, metadata, files, vision_prefix, device)
    processor = AutoProcessor.from_pretrained(
        args.model,
        revision=args.revision,
        local_files_only=args.local_files_only,
    )
    if args.image:
        with Image.open(args.image) as opened:
            image = opened.convert("RGB")
        source = str(args.image)
    else:
        image = Image.new("RGB", (args.synthetic_size, args.synthetic_size), color="white")
        source = f"synthetic {args.synthetic_size}x{args.synthetic_size} RGB image"

    inputs = processor.image_processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(
        device=device, dtype=next(vision.parameters()).dtype
    )
    grid = inputs["image_grid_thw"].to(device=device)

    captures: dict[str, Any] = {}
    hook = vision.merger.register_forward_hook(
        lambda _module, hook_inputs, output: captures.update(
            projector_input=describe_value(hook_inputs[0]),
            projector_output=describe_value(output),
        )
    )
    try:
        with torch.inference_mode():
            output = vision(hidden_states=pixel_values, grid_thw=grid)
    finally:
        hook.remove()

    return {
        "image": source,
        "device": device,
        "processed_pixels": describe_value(pixel_values),
        "image_grid_thw": grid.cpu().tolist(),
        "encoder_output": describe_value(output.last_hidden_state),
        **captures,
    }


def format_parameters(value: int) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.3f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.3f}M"
    if value >= 1_000:
        return f"{value / 1_000:.3f}K"
    return str(value)


def print_report(report: dict[str, Any]) -> None:
    print(f"Model: {report['model']}")
    print(f"Architecture: {report['architecture']}")
    print("\nVision configuration:")
    for name, value in report["vision_config"].items():
        print(f"  {name}: {value}")
    for title, key in (
        ("Vision tower", "vision_tower"),
        ("Encoder (projector excluded)", "encoder"),
        ("Projector", "projector"),
    ):
        item = report.get(key)
        if item:
            print(
                f"\n{title}: {item['prefix']}\n"
                f"  parameters: {item['parameters']:,} ({format_parameters(item['parameters'])})\n"
                f"  tensors: {item['tensor_count']}\n"
                f"  dtypes: {item['dtypes']}"
            )
    if report.get("largest_vision_tensors"):
        print("\nLargest vision tensors:")
        for item in report["largest_vision_tensors"]:
            print(f"  {item['name']}: {item['shape']} {item['dtype']} ({format_parameters(item['parameters'])})")
    if report.get("forward"):
        print("\nForward pass:")
        for name, value in report["forward"].items():
            print(f"  {name}: {value}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        args.model,
        revision=args.revision,
        local_files_only=args.local_files_only,
    )
    report: dict[str, Any] = {
        "model": args.model,
        "architecture": (config.architectures or [type(config).__name__])[0],
        "vision_config": selected_vision_config(config),
    }

    files: list[Path] = []
    metadata: dict[str, dict[str, Any]] = {}
    vision_prefix = projector_prefix = None
    if not args.config_only:
        files = locate_safetensors(args.model, args.revision, args.local_files_only)
        metadata = tensor_metadata(files)
        names = list(metadata)
        vision_prefix = find_vision_prefix(names)
        projector_prefix = find_projector_prefix(names, vision_prefix)
        if vision_prefix is None:
            raise RuntimeError("Could not identify a vision tower in the checkpoint")

        report["checkpoint_files"] = [str(path) for path in files]
        report["vision_tower"] = parameter_summary(metadata, vision_prefix)
        report["projector"] = parameter_summary(metadata, projector_prefix)
        encoder_names = {
            name: info
            for name, info in metadata.items()
            if name.startswith(vision_prefix)
            and (projector_prefix is None or not name.startswith(projector_prefix))
        }
        encoder_summary = parameter_summary(encoder_names, vision_prefix)
        report["encoder"] = encoder_summary

        largest = sorted(
            (
                {"name": name, **info}
                for name, info in metadata.items()
                if name.startswith(vision_prefix)
            ),
            key=lambda item: item["parameters"],
            reverse=True,
        )[: args.show_tensors]
        for item in largest:
            item.pop("file", None)
        report["largest_vision_tensors"] = largest

    if args.run_forward:
        assert vision_prefix is not None
        report["forward"] = run_forward(args, config, metadata, files, vision_prefix)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report)


if __name__ == "__main__":
    main()
