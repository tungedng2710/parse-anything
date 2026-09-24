from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch
from PIL import Image, ImageFile
from torch.utils.data import IterableDataset, get_worker_info

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
OVERLENGTH_COUNT_KEY = "_overlength_count"
ORIGINAL_COUNT_KEY = "_original_count"


def _stable_fraction(value: str) -> float:
    digest = hashlib.sha1(value.encode("utf-8"), usedforsecurity=False).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _stable_integer(value: str) -> int:
    digest = hashlib.sha1(value.encode("utf-8"), usedforsecurity=False).digest()
    return int.from_bytes(digest[8:16], "big")


class PairedJsonDataset(IterableDataset[dict[str, Any]]):
    """Stream images paired with JSON labels by relative filename stem."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        images_subdir: str = "images",
        labels_subdir: str = "labels",
        validation_fraction: float = 0.02,
        document_id_regex: str = r"_page\d+$",
        seed: int = 42,
        shuffle_buffer: int = 2_000,
        process_index: int = 0,
        num_processes: int = 1,
    ) -> None:
        super().__init__()
        if split not in {"train", "validation"}:
            raise ValueError("split must be 'train' or 'validation'")
        if not 0.0 < validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between 0 and 1")
        if shuffle_buffer <= 0:
            raise ValueError("shuffle_buffer must be positive")
        if num_processes <= 0 or not 0 <= process_index < num_processes:
            raise ValueError("process_index must identify one of num_processes ranks")

        self.root = Path(root).expanduser()
        self.images_root = self.root / images_subdir
        self.labels_root = self.root / labels_subdir
        if not self.images_root.is_dir():
            raise FileNotFoundError(
                f"Image directory does not exist: {self.images_root}"
            )
        if not self.labels_root.is_dir():
            raise FileNotFoundError(
                f"Label directory does not exist: {self.labels_root}"
            )

        self.split = split
        self.validation_fraction = validation_fraction
        self.document_pattern = re.compile(document_id_regex)
        self.seed = seed
        self.shuffle_buffer = shuffle_buffer
        self.process_index = process_index
        self.num_processes = num_processes
        self._epoch = 0

    def _image_paths(self) -> Iterator[Path]:
        yield from (
            path
            for path in self.images_root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )

    def _relative_key(self, image_path: Path) -> str:
        return image_path.relative_to(self.images_root).with_suffix("").as_posix()

    def _document_id(self, relative_key: str) -> str:
        return self.document_pattern.sub("", relative_key)

    def _belongs_to_split(self, relative_key: str) -> bool:
        validation = (
            _stable_fraction(self._document_id(relative_key)) < self.validation_fraction
        )
        return validation if self.split == "validation" else not validation

    def _label_path(self, image_path: Path) -> Path:
        relative = image_path.relative_to(self.images_root)
        return (self.labels_root / relative).with_suffix(".json")

    def _load(self, image_path: Path) -> dict[str, Any]:
        label_path = self._label_path(image_path)
        if not label_path.is_file():
            raise FileNotFoundError(
                f"Missing JSON label for {image_path}: {label_path}"
            )
        try:
            label = json.loads(label_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"Could not read valid JSON from {label_path}"
            ) from error
        try:
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
        except (OSError, ValueError) as error:
            raise RuntimeError(f"Could not decode image {image_path}") from error
        target = json.dumps(
            label, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return {"image": image, "target": target, "source": str(image_path)}

    def __iter__(self) -> Iterator[dict[str, Any]]:
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        worker_count = worker.num_workers if worker else 1
        global_worker_id = self.process_index * worker_count + worker_id
        global_worker_count = self.num_processes * worker_count
        rng = random.Random(self.seed + self._epoch * 1_000_003 + global_worker_id)
        self._epoch += 1

        buffer: list[Path] = []
        for image_path in self._image_paths():
            relative_key = self._relative_key(image_path)
            if not self._belongs_to_split(relative_key):
                continue
            if _stable_integer(relative_key) % global_worker_count != global_worker_id:
                continue
            if len(buffer) < self.shuffle_buffer:
                buffer.append(image_path)
                continue
            selected = rng.randrange(len(buffer))
            candidate, buffer[selected] = buffer[selected], image_path
            yield self._load(candidate)

        rng.shuffle(buffer)
        for image_path in buffer:
            yield self._load(image_path)


def find_subsequence(sequence: Sequence[int], subsequence: Sequence[int]) -> int:
    """Return the final occurrence of a token subsequence."""
    width = len(subsequence)
    for index in range(len(sequence) - width, -1, -1):
        if list(sequence[index : index + width]) == list(subsequence):
            return index
    return -1


class AlignmentCollator:
    """Build Chandra multimodal inputs with assistant-only JSON labels."""

    def __init__(
        self,
        processor: Any,
        prompt: str,
        min_pixels: int = 65_536,
        max_pixels: int = 589_824,
        max_sequence_length: int = 8_192,
        overlength_policy: str = "skip",
    ) -> None:
        self.processor = processor
        self.prompt = prompt
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.max_sequence_length = max_sequence_length
        if overlength_policy not in {"skip", "error"}:
            raise ValueError("overlength_policy must be 'skip' or 'error'")
        self.overlength_policy = overlength_policy
        marker = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.target_marker_ids = processor.tokenizer(
            marker, add_special_tokens=False
        ).input_ids
        self.end_token_id = processor.tokenizer.eos_token_id
        if not self.target_marker_ids:
            raise RuntimeError("Could not tokenize the assistant target marker")

    def _conversation(self, target: str) -> list[dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.prompt},
                ],
            },
            {"role": "assistant", "content": target},
        ]

    def _process(self, examples: Sequence[dict[str, Any]]) -> Any:
        texts = [
            self.processor.apply_chat_template(
                self._conversation(example["target"]),
                tokenize=False,
                add_generation_prompt=False,
            )
            for example in examples
        ]
        return self.processor(
            images=[example["image"] for example in examples],
            text=texts,
            padding=True,
            truncation=True,
            max_length=self.max_sequence_length,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
            return_tensors="pt",
        )

    def _invalid_rows(self, inputs: Any) -> list[int]:
        invalid: list[int] = []
        for row in range(inputs["input_ids"].shape[0]):
            token_ids = inputs["input_ids"][row].tolist()
            marker_start = find_subsequence(token_ids, self.target_marker_ids)
            if marker_start < 0:
                invalid.append(row)
                continue
            target_start = marker_start + len(self.target_marker_ids)
            target_ids = inputs["input_ids"][row, target_start:]
            target_mask = inputs["attention_mask"][row, target_start:].bool()
            if self.end_token_id not in target_ids[target_mask].tolist():
                invalid.append(row)
        return invalid

    def _make_labels(self, inputs: Any) -> torch.Tensor:
        labels = inputs["input_ids"].clone()
        for row in range(labels.shape[0]):
            token_ids = inputs["input_ids"][row].tolist()
            marker_start = find_subsequence(token_ids, self.target_marker_ids)
            if marker_start < 0:
                raise RuntimeError(
                    "Assistant target marker was truncated or missing; increase max_sequence_length"
                )
            target_start = marker_start + len(self.target_marker_ids)
            labels[row, :target_start] = -100
            labels[row, inputs["attention_mask"][row] == 0] = -100
            if not torch.any(labels[row] != -100):
                raise RuntimeError(
                    "No assistant target tokens remain after truncation; increase max_sequence_length"
                )
            if self.end_token_id not in labels[row][labels[row] != -100].tolist():
                raise AssertionError("An overlength row survived collator filtering")
        return labels

    def __call__(self, examples: Sequence[dict[str, Any]]) -> dict[str, Any]:
        original_count = len(examples)
        inputs = self._process(examples)
        invalid_rows = self._invalid_rows(inputs)
        if invalid_rows and self.overlength_policy == "error":
            sources = [
                str(examples[index].get("source", f"batch row {index}"))
                for index in invalid_rows[:3]
            ]
            raise RuntimeError(
                f"{len(invalid_rows)} JSON target(s) exceeded max_sequence_length="
                f"{self.max_sequence_length}; first: {', '.join(sources)}"
            )

        if invalid_rows:
            invalid = set(invalid_rows)
            examples = [
                example
                for index, example in enumerate(examples)
                if index not in invalid
            ]
            if not examples:
                return {
                    OVERLENGTH_COUNT_KEY: torch.tensor(len(invalid_rows)),
                    ORIGINAL_COUNT_KEY: torch.tensor(original_count),
                }
            # Multimodal pixel tensors are packed rather than batch-indexed, so
            # rebuild the processor output instead of slicing the original batch.
            inputs = self._process(examples)

        inputs["labels"] = self._make_labels(inputs)
        inputs[OVERLENGTH_COUNT_KEY] = torch.tensor(len(invalid_rows))
        inputs[ORIGINAL_COUNT_KEY] = torch.tensor(original_count)
        return dict(inputs)
