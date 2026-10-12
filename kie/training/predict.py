"""Generate local GPU predictions for the baseline or a saved TT32 adapter."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

from .common import dump_json, load_json, read_manifest, validate_label
from .modeling import load_base, load_images, load_processor, prompt_text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/tt32/test.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, help="Final adapter directory from training")
    parser.add_argument("--model", default="numind/NuExtract3")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--input-mode", choices=["image", "image-text"], default="image")
    parser.add_argument("--no-field-instructions", action="store_true",
                        help="Types-only B0 baseline; default includes current field descriptions (B1)")
    parser.add_argument("--max-pixels", type=int, default=1605632)
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--qlora", action="store_true", help="Load the base in NF4 on supported hardware")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if min(args.max_pixels, args.max_length, args.max_new_tokens) <= 0:
        parser.error("Pixel and token limits must be positive")
    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        parser.error("Prediction directory is nonempty; choose a new one or --overwrite")
    rows = read_manifest(args.manifest)
    if args.adapter:
        config_path = args.adapter.parent / "run_config.json"
        if not config_path.is_file():
            parser.error("Adapter requires the adjacent run_config.json from training")
        config = load_json(config_path)
        args.model = config["model"]
        args.revision = config.get("resolved_model_revision") or config["revision"]
        args.input_mode, args.max_pixels = config["input_mode"], config["max_pixels"]
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required. Prediction has not been started.")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    processor = load_processor(args.model, args.revision, args.max_pixels)
    if args.adapter:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(args.adapter)
        processor.image_processor.size["longest_edge"] = args.max_pixels
    model = load_base(args.model, args.revision, dtype, args.qlora, "sdpa")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    if not args.qlora:
        model.to("cuda")
    model.eval()
    metadata = []
    for row in rows:
        images = load_images(row)
        text = prompt_text(processor, row, images, args.input_mode,
                           include_instructions=not args.no_field_instructions)
        inputs = processor(text=[text], images=images, truncation=False,
                           add_special_tokens=False, return_tensors="pt").to("cuda")
        prompt_length = inputs["input_ids"].shape[1]
        if prompt_length + args.max_new_tokens > args.max_length:
            raise ValueError(f"{row['id']}: prompt plus generation budget exceeds --max-length")
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        continuation = output[0, prompt_length:]
        raw = processor.tokenizer.decode(continuation, skip_special_tokens=True).strip()
        if "</think>" in raw:
            raw = raw.split("</think>", 1)[1].strip()
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / f"{row['id']}.txt").write_text(raw, encoding="utf-8")
        error = None
        parsed = False
        try:
            import json
            prediction = json.loads(raw)
            dump_json(args.output / f"{row['id']}.json", prediction)
            parsed = True
            validate_label(prediction, row["template"])
        except ValueError as exc:
            error = str(exc)
            if not parsed:
                # Remove any stale valid result if --overwrite reruns fail JSON parsing.
                (args.output / f"{row['id']}.json").unlink(missing_ok=True)
        metadata.append({"id": row["id"], "seconds": elapsed,
                         "prompt_tokens": prompt_length, "generated_tokens": len(continuation),
                         "hit_token_limit": len(continuation) >= args.max_new_tokens,
                         "json_valid": parsed, "schema_valid": parsed and error is None, "error": error})
        print(f"{row['id']}: {elapsed:.2f}s, {len(continuation)} output tokens")
    dump_json(args.output / "generation_metadata.json", {"documents": metadata,
              "model": args.model, "revision": args.revision, "input_mode": args.input_mode,
              "max_pixels": args.max_pixels, "field_instructions": not args.no_field_instructions,
              "adapter": str(args.adapter) if args.adapter else None})


if __name__ == "__main__":
    main()
