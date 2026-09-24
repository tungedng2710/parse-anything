#!/usr/bin/env python3
"""Run the final Chandra SFT model on one document image."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from chandra_alignment.metrics import extract_json
from chandra_mae.checkpoint import apply_vision_delta


@dataclass(frozen=True)
class SFTArtifacts:
    base_model: str
    mae_delta: Path | None
    alignment_delta: Path | None
    merger_delta: Path
    lora_adapter: Path
    processor: Path | str


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required manifest does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_path(value: str | Path, root: Path) -> Path:
    """Resolve paths saved either relative to the run directory or launch cwd."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    return (root / path).resolve()


def resolve_sft_artifacts(
    sft_dir: str | Path,
    *,
    base_model: str | None = None,
    mae_delta: str | Path | None = None,
    alignment_delta: str | Path | None = None,
    merger_delta: str | Path | None = None,
    lora_adapter: str | Path | None = None,
    processor: str | Path | None = None,
) -> SFTArtifacts:
    run_dir = Path(sft_dir).expanduser().resolve()
    manifest = _read_json(run_dir / "sft_manifest.json")

    resolved_base = base_model or manifest.get("base_model")
    if not resolved_base:
        raise ValueError("base_model is absent from the SFT manifest")

    def optional_path(override: str | Path | None, key: str) -> Path | None:
        value = override if override is not None else manifest.get(key)
        return _resolve_path(value, run_dir) if value else None

    resolved_mae = optional_path(mae_delta, "mae_delta")
    resolved_alignment = optional_path(alignment_delta, "alignment_delta")
    resolved_merger = _resolve_path(
        merger_delta or manifest.get("merger_delta", "sft_merger_delta.safetensors"),
        run_dir,
    )
    resolved_adapter = _resolve_path(
        lora_adapter or manifest.get("lora_adapter", "lora_adapter"), run_dir
    )
    resolved_processor: Path | str
    if processor is None:
        saved_processor = run_dir / "processor"
        resolved_processor = saved_processor if saved_processor.is_dir() else resolved_base
    else:
        processor_path = Path(processor).expanduser()
        resolved_processor = (
            _resolve_path(processor_path, run_dir)
            if processor_path.is_absolute() or processor_path.exists()
            else str(processor)
        )

    required = {
        "SFT merger delta": resolved_merger,
        "LoRA adapter": resolved_adapter,
    }
    if resolved_mae is not None:
        required["MAE delta"] = resolved_mae
    if resolved_alignment is not None:
        required["alignment delta"] = resolved_alignment
    for label, path in required.items():
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")

    return SFTArtifacts(
        base_model=resolved_base,
        mae_delta=resolved_mae,
        alignment_delta=resolved_alignment,
        merger_delta=resolved_merger,
        lora_adapter=resolved_adapter,
        processor=resolved_processor,
    )


def load_sft_model(
    artifacts: SFTArtifacts,
    *,
    device: torch.device,
    dtype: torch.dtype,
    attention_implementation: str,
    local_files_only: bool,
) -> tuple[torch.nn.Module, Any]:
    processor = AutoProcessor.from_pretrained(
        artifacts.processor, local_files_only=local_files_only
    )
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"

    model = AutoModelForImageTextToText.from_pretrained(
        artifacts.base_model,
        dtype=dtype,
        attn_implementation=attention_implementation,
        local_files_only=local_files_only,
    )
    if artifacts.mae_delta is not None:
        apply_vision_delta(model, artifacts.mae_delta)
    if artifacts.alignment_delta is not None:
        apply_vision_delta(model, artifacts.alignment_delta)
    apply_vision_delta(model, artifacts.merger_delta)
    model = PeftModel.from_pretrained(model, artifacts.lora_adapter)
    model.to(device).eval()
    return model, processor


def generate_single(
    model: torch.nn.Module,
    processor: Any,
    image: Image.Image,
    prompt: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
    min_pixels: int,
    max_pixels: int,
    max_input_length: int,
    max_new_tokens: int,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
) -> tuple[str, int, float]:
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
    inputs = processor(
        images=[image],
        text=[text],
        padding=True,
        truncation=True,
        max_length=max_input_length,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    prompt_length = inputs["input_ids"].shape[1]
    tokenizer_eos = processor.tokenizer.eos_token_id
    configured_eos = getattr(model.generation_config, "eos_token_id", None)
    eos_token_ids = (
        list(configured_eos) if isinstance(configured_eos, list) else [configured_eos]
    )
    if tokenizer_eos not in eos_token_ids:
        eos_token_ids.append(tokenizer_eos)
    eos_token_ids = [token for token in eos_token_ids if token is not None]
    started = time.perf_counter()
    with (
        torch.inference_mode(),
        torch.autocast(
            device_type=device.type,
            dtype=dtype,
            enabled=device.type == "cuda" and dtype != torch.float32,
        ),
    ):
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            eos_token_id=eos_token_ids,
            pad_token_id=processor.tokenizer.pad_token_id,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
    elapsed = time.perf_counter() - started
    generated_tokens = generated.shape[1] - prompt_length
    response = processor.batch_decode(
        generated[:, prompt_length:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return response, generated_tokens, elapsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Document image to process")
    parser.add_argument("--sft-dir", type=Path, default=Path("outputs/chandra2-sft"))
    parser.add_argument(
        "--prompt", default="Extract the document as JSON. Return JSON only."
    )
    parser.add_argument("--output", type=Path, help="Write the inference record as JSON")
    parser.add_argument("--base-model")
    parser.add_argument("--mae-delta", type=Path)
    parser.add_argument("--alignment-delta", type=Path)
    parser.add_argument("--merger-delta", type=Path)
    parser.add_argument("--lora-adapter", type=Path)
    parser.add_argument("--processor")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mixed-precision", choices=("bf16", "fp16", "no"), default="bf16")
    parser.add_argument("--attention-implementation", default="sdpa")
    parser.add_argument("--min-pixels", type=int, default=65_536)
    parser.add_argument("--max-pixels", type=int, default=589_824)
    parser.add_argument("--max-input-length", type=int, default=8_192)
    parser.add_argument("--max-new-tokens", type=int, default=8_192)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--require-valid-json",
        action="store_true",
        help="Exit with status 2 when generation is not one complete JSON value",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(f"Image does not exist: {args.image}")
    if min(args.min_pixels, args.max_pixels, args.max_input_length, args.max_new_tokens) <= 0:
        raise ValueError("Pixel and token limits must be positive")
    if args.min_pixels > args.max_pixels:
        raise ValueError("min_pixels cannot exceed max_pixels")
    if args.repetition_penalty <= 0 or args.no_repeat_ngram_size < 0:
        raise ValueError("Invalid repetition controls")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available")
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "no": torch.float32,
    }[args.mixed_precision]
    artifacts = resolve_sft_artifacts(
        args.sft_dir,
        base_model=args.base_model,
        mae_delta=args.mae_delta,
        alignment_delta=args.alignment_delta,
        merger_delta=args.merger_delta,
        lora_adapter=args.lora_adapter,
        processor=args.processor,
    )

    print(f"Loading base model: {artifacts.base_model}", file=sys.stderr)
    print(f"Loading SFT run: {Path(args.sft_dir).resolve()}", file=sys.stderr)
    model, processor = load_sft_model(
        artifacts,
        device=device,
        dtype=dtype,
        attention_implementation=args.attention_implementation,
        local_files_only=args.local_files_only,
    )
    with Image.open(args.image) as opened:
        image = opened.convert("RGB")
    response, token_count, elapsed = generate_single(
        model,
        processor,
        image,
        args.prompt,
        device=device,
        dtype=dtype,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        max_input_length=args.max_input_length,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )
    parsed = extract_json(response)
    record = {
        "image": str(args.image.resolve()),
        "prompt": args.prompt,
        "json_valid": parsed is not None,
        "generated_tokens": token_count,
        "elapsed_seconds": elapsed,
        "tokens_per_second": token_count / elapsed if elapsed else None,
        "parsed_prediction": parsed,
        "raw_response": response,
    }
    rendered = json.dumps(record, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Saved inference record to {args.output}", file=sys.stderr)
    if args.require_valid_json and parsed is None:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
