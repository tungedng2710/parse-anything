from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset, IterableDataset, get_worker_info

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_SUFFIXES = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def unpatchify_chandra_image(
    patches: torch.Tensor,
    grid_thw: torch.Tensor,
    patch_size: int,
    temporal_patch_size: int,
    spatial_merge_size: int,
    num_channels: int = 3,
) -> torch.Tensor:
    """Invert Qwen/Chandra image patch ordering into ``[T, C, H, W]`` pixels."""
    grid_t, grid_h, grid_w = (int(value) for value in grid_thw.tolist())
    merge = spatial_merge_size
    expected = grid_t * grid_h * grid_w
    if patches.shape != (
        expected,
        num_channels * temporal_patch_size * patch_size * patch_size,
    ):
        raise ValueError(
            f"Expected {(expected, num_channels * temporal_patch_size * patch_size**2)} "
            f"patch values, received {tuple(patches.shape)}"
        )
    if grid_h % merge or grid_w % merge:
        raise ValueError(
            "Spatial grid dimensions must be divisible by spatial_merge_size"
        )
    arranged = patches.reshape(
        grid_t,
        grid_h // merge,
        grid_w // merge,
        merge,
        merge,
        num_channels,
        temporal_patch_size,
        patch_size,
        patch_size,
    )
    return arranged.permute(0, 6, 5, 1, 3, 7, 2, 4, 8).reshape(
        grid_t * temporal_patch_size,
        num_channels,
        grid_h * patch_size,
        grid_w * patch_size,
    )


def discover_images(root: str | Path, recursive: bool = True) -> list[Path]:
    root = Path(root).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {root}")
    iterator = root.rglob("*") if recursive else root.glob("*")
    paths = sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise ValueError(f"No supported images found in {root}")
    return paths


class DocumentImageDataset(Dataset[dict[str, Any]]):
    """Path-backed image dataset with bounded recovery from corrupt files."""

    def __init__(
        self, root: str | Path, recursive: bool = True, seed: int = 42
    ) -> None:
        self.paths = discover_images(root, recursive=recursive)
        self.seed = seed

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        # A bad page should not terminate a multi-million-page run. Try a stable,
        # worker-independent fallback sequence and still surface persistent errors.
        rng = random.Random(self.seed + index)
        candidates = [index] + [rng.randrange(len(self.paths)) for _ in range(9)]
        last_error: Exception | None = None
        for candidate in candidates:
            path = self.paths[candidate]
            try:
                with Image.open(path) as opened:
                    image = opened.convert("RGB")
                return {"image": image, "path": str(path)}
            except (OSError, ValueError) as error:
                last_error = error
        raise RuntimeError(
            f"Could not decode an image after 10 attempts near index {index}"
        ) from last_error


class StreamingDocumentImageDataset(IterableDataset[dict[str, Any]]):
    """Bounded-memory directory stream with approximate buffer shuffling.

    This is the production path for the ~20M-page corpus. Files are assigned
    disjointly across DataLoader workers; Accelerate subsequently shards batches
    across distributed processes.
    """

    def __init__(
        self,
        root: str | Path,
        recursive: bool = True,
        seed: int = 42,
        shuffle_buffer: int = 10_000,
        process_index: int = 0,
        num_processes: int = 1,
    ) -> None:
        super().__init__()
        self.root = Path(root).expanduser()
        if not self.root.is_dir():
            raise FileNotFoundError(f"Image directory does not exist: {self.root}")
        if shuffle_buffer <= 0:
            raise ValueError("shuffle_buffer must be positive")
        if num_processes <= 0 or not 0 <= process_index < num_processes:
            raise ValueError("process_index must identify one of num_processes ranks")
        self.recursive = recursive
        self.seed = seed
        self.shuffle_buffer = shuffle_buffer
        self.process_index = process_index
        self.num_processes = num_processes
        self._epoch = 0

    def _paths(self):
        iterator = self.root.rglob("*") if self.recursive else self.root.glob("*")
        yield from (
            path
            for path in iterator
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )

    @staticmethod
    def _decode(path: Path) -> dict[str, Any] | None:
        try:
            with Image.open(path) as opened:
                image = opened.convert("RGB")
            return {"image": image, "path": str(path)}
        except (OSError, ValueError):
            return None

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        worker_count = worker.num_workers if worker else 1
        global_worker_id = self.process_index * worker_count + worker_id
        global_worker_count = self.num_processes * worker_count
        rng = random.Random(self.seed + self._epoch * 1_000_003 + global_worker_id)
        self._epoch += 1
        buffer: list[Path] = []
        for index, path in enumerate(self._paths()):
            if index % global_worker_count != global_worker_id:
                continue
            if len(buffer) < self.shuffle_buffer:
                buffer.append(path)
                continue
            selected = rng.randrange(len(buffer))
            candidate, buffer[selected] = buffer[selected], path
            decoded = self._decode(candidate)
            if decoded is not None:
                yield decoded
        rng.shuffle(buffer)
        for path in buffer:
            decoded = self._decode(path)
            if decoded is not None:
                yield decoded


class ChandraImageCollator:
    """Apply Chandra's native preprocessing and preserve raw RGB patch targets."""

    def __init__(
        self,
        image_processor: Any,
        min_pixels: int = 256 * 256,
        max_pixels: int = 768 * 768,
    ) -> None:
        self.image_processor = image_processor
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.patch_size = int(image_processor.patch_size)
        self.temporal_patch_size = int(image_processor.temporal_patch_size)
        self.num_channels = len(image_processor.image_mean)
        self.mean = torch.tensor(image_processor.image_mean, dtype=torch.float32)
        self.std = torch.tensor(image_processor.image_std, dtype=torch.float32)

    def _to_raw_rgb(self, normalized: torch.Tensor) -> torch.Tensor:
        patches = normalized.float().reshape(
            -1,
            self.num_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        mean = self.mean.view(1, -1, 1, 1, 1)
        std = self.std.view(1, -1, 1, 1, 1)
        return (patches * std + mean).clamp_(0.0, 1.0).flatten(1)

    def __call__(self, examples: Sequence[dict[str, Any]]) -> dict[str, Any]:
        encoded = self.image_processor(
            images=[example["image"] for example in examples],
            return_tensors="pt",
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        pixel_values = encoded["pixel_values"]
        grid_thw = encoded["image_grid_thw"]
        expected = int(grid_thw.prod(dim=1).sum().item())
        if expected != pixel_values.shape[0]:
            raise RuntimeError(
                f"Processor returned {pixel_values.shape[0]} patches, but grid_thw describes {expected}"
            )
        return {
            "pixel_values": pixel_values,
            "target_patches": self._to_raw_rgb(pixel_values),
            "grid_thw": grid_thw,
        }
