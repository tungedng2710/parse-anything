#!/usr/bin/env python3
"""Copy a uniform reservoir sample of images without loading all paths into RAM."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from chandra_mae.data import IMAGE_SUFFIXES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--count", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--recursive", action=argparse.BooleanOptionalAction, default=False
    )
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("--count must be positive")
    if not args.source.is_dir():
        parser.error(f"source is not a directory: {args.source}")
    args.destination.mkdir(parents=True, exist_ok=True)
    if any(args.destination.iterdir()):
        parser.error(f"destination must be empty: {args.destination}")

    rng = random.Random(args.seed)
    sample: list[Path] = []
    seen = 0
    iterator = args.source.rglob("*") if args.recursive else args.source.glob("*")
    for path in iterator:
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        seen += 1
        if len(sample) < args.count:
            sample.append(path)
        else:
            replacement = rng.randrange(seen)
            if replacement < args.count:
                sample[replacement] = path
    if len(sample) < args.count:
        parser.error(f"source contains only {len(sample)} supported images")
    for index, source in enumerate(sample, start=1):
        shutil.copy2(source, args.destination / source.name)
        if index % 1000 == 0:
            print(f"Copied {index:,}/{args.count:,}", flush=True)


if __name__ == "__main__":
    main()
