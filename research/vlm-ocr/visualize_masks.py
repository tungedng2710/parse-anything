#!/usr/bin/env python3
"""Render deterministic Chandra patch masks for a small document sample."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor

from chandra_mae.data import (
    ChandraImageCollator,
    DocumentImageDataset,
    unpatchify_chandra_image,
)
from chandra_mae.model import random_patch_mask


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    pixels = tensor.detach().clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(pixels)


def fit_panel(image: Image.Image, width: int, height: int) -> Image.Image:
    copy = image.copy()
    copy.thumbnail((width, height), Image.Resampling.LANCZOS)
    panel = Image.new("RGB", (width, height), "#d8d8d8")
    panel.paste(copy, ((width - copy.width) // 2, (height - copy.height) // 2))
    return panel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("sample_dataset"))
    parser.add_argument("--model", default="datalab-to/chandra-ocr-2")
    parser.add_argument(
        "--output", type=Path, default=Path("previews/masked_samples.png")
    )
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mask-ratio", type=float, default=0.75)
    parser.add_argument("--min-pixels", type=int, default=65_536)
    parser.add_argument("--max-pixels", type=int, default=589_824)
    parser.add_argument("--panel-width", type=int, default=560)
    parser.add_argument("--panel-height", type=int, default=440)
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("--count must be positive")

    dataset = DocumentImageDataset(args.dataset, recursive=True, seed=args.seed)
    if args.count > len(dataset):
        parser.error(
            f"requested {args.count} images from a {len(dataset)}-image dataset"
        )
    indices = random.Random(args.seed).sample(range(len(dataset)), args.count)
    examples = [dataset[index] for index in indices]
    processor = AutoProcessor.from_pretrained(args.model)
    collator = ChandraImageCollator(
        processor.image_processor,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    batch = collator(examples)
    lengths = [int(value) for value in batch["grid_thw"].prod(dim=1).tolist()]
    generator = torch.Generator().manual_seed(args.seed)
    masks = random_patch_mask(lengths, args.mask_ratio, generator=generator)

    font = ImageFont.load_default(size=18)
    header_height = 34
    sheet = Image.new(
        "RGB",
        (args.panel_width * 2, (args.panel_height + header_height) * args.count),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    offset = 0
    patch_size = int(processor.image_processor.patch_size)
    temporal_patch_size = int(processor.image_processor.temporal_patch_size)
    merge_size = int(processor.image_processor.merge_size)
    for row, (example, length, grid) in enumerate(
        zip(examples, lengths, batch["grid_thw"], strict=True)
    ):
        raw = batch["target_patches"][offset : offset + length]
        sample_mask = masks[offset : offset + length]
        masked = raw.clone()
        masked[sample_mask] = 0.5
        original_image = tensor_to_image(
            unpatchify_chandra_image(
                raw, grid, patch_size, temporal_patch_size, merge_size
            )[0]
        )
        masked_image = tensor_to_image(
            unpatchify_chandra_image(
                masked, grid, patch_size, temporal_patch_size, merge_size
            )[0]
        )
        y = row * (args.panel_height + header_height)
        name = Path(example["path"]).name
        if len(name) > 90:
            name = name[:87] + "..."
        draw.text((8, y + 7), name, fill="black", font=font)
        sheet.paste(
            fit_panel(original_image, args.panel_width, args.panel_height),
            (0, y + header_height),
        )
        sheet.paste(
            fit_panel(masked_image, args.panel_width, args.panel_height),
            (args.panel_width, y + header_height),
        )
        draw.text(
            (10, y + header_height + 8), "processed image", fill="#c02020", font=font
        )
        draw.text(
            (args.panel_width + 10, y + header_height + 8),
            f"{args.mask_ratio:.0%} patches masked",
            fill="#c02020",
            font=font,
        )
        offset += length

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)
    print(f"Saved {args.count} masked examples to {args.output}")


if __name__ == "__main__":
    main()
