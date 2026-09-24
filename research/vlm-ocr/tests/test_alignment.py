from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from PIL import Image
from transformers import AutoProcessor

from chandra_alignment.data import (
    ORIGINAL_COUNT_KEY,
    OVERLENGTH_COUNT_KEY,
    AlignmentCollator,
    PairedJsonDataset,
)
from train_alignment import configure_trainable_parameters, save_alignment_delta


def make_pair(root: Path, stem: str, label: dict) -> None:
    image_path = root / "images" / f"{stem}.png"
    label_path = root / "labels" / f"{stem}.json"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 96), "white").save(image_path)
    label_path.write_text(json.dumps(label), encoding="utf-8")


def test_paired_dataset_split_grouping_and_rank_sharding(tmp_path):
    for document in range(20):
        for page in range(2):
            make_pair(
                tmp_path,
                f"nested/doc{document}_page{page}",
                {"document": document, "page": page},
            )

    train = list(
        PairedJsonDataset(
            tmp_path,
            split="train",
            validation_fraction=0.25,
            shuffle_buffer=2,
        )
    )
    validation = list(
        PairedJsonDataset(
            tmp_path,
            split="validation",
            validation_fraction=0.25,
            shuffle_buffer=2,
        )
    )
    assert len(train) + len(validation) == 40
    train_documents = {json.loads(example["target"])["document"] for example in train}
    validation_documents = {
        json.loads(example["target"])["document"] for example in validation
    }
    assert train_documents.isdisjoint(validation_documents)

    rank_zero = list(
        PairedJsonDataset(
            tmp_path,
            split="train",
            validation_fraction=0.25,
            shuffle_buffer=2,
            process_index=0,
            num_processes=2,
        )
    )
    rank_one = list(
        PairedJsonDataset(
            tmp_path,
            split="train",
            validation_fraction=0.25,
            shuffle_buffer=2,
            process_index=1,
            num_processes=2,
        )
    )
    assert len(rank_zero) + len(rank_one) == len(train)
    assert all(example["target"].startswith('{"document":') for example in train)


def test_missing_label_is_reported(tmp_path):
    image_dir = tmp_path / "images"
    (tmp_path / "labels").mkdir()
    image_dir.mkdir()
    Image.new("RGB", (16, 16), "white").save(image_dir / "missing.png")
    dataset = PairedJsonDataset(
        tmp_path,
        split="train",
        validation_fraction=0.01,
        shuffle_buffer=1,
    )
    with pytest.raises(FileNotFoundError, match="Missing JSON label"):
        list(dataset)


@pytest.fixture(scope="module")
def processor():
    return AutoProcessor.from_pretrained(
        "datalab-to/chandra-ocr-2", local_files_only=True
    )


def test_collator_masks_everything_before_json_target(processor):
    collator = AlignmentCollator(
        processor,
        prompt="Extract JSON.",
        min_pixels=65_536,
        max_pixels=65_536,
        max_sequence_length=512,
    )
    batch = collator(
        [
            {
                "image": Image.new("RGB", (64, 96), "white"),
                "target": '{"amount":123,"valid":true}',
            }
        ]
    )
    labels = batch["labels"][0]
    target_positions = torch.where(labels != -100)[0]
    assert target_positions.numel() > 0
    assert torch.all(labels[: target_positions[0]] == -100)
    decoded = processor.tokenizer.decode(labels[target_positions].tolist())
    assert '"amount":123' in decoded
    assert "<|im_end|>" in decoded
    assert batch["pixel_values"].shape[1] == 1_536
    assert batch[OVERLENGTH_COUNT_KEY].item() == 0
    assert batch[ORIGINAL_COUNT_KEY].item() == 1


def test_collator_skips_overlength_sample_and_rebuilds_batch(processor):
    collator = AlignmentCollator(
        processor,
        prompt="Extract JSON.",
        min_pixels=65_536,
        max_pixels=65_536,
        max_sequence_length=512,
        overlength_policy="skip",
    )
    image = Image.new("RGB", (64, 96), "white")
    short = {"image": image, "target": '{"ok":true}', "source": "short.png"}
    long = {
        "image": image,
        "target": json.dumps({"text": "word " * 2_000}),
        "source": "long.png",
    }
    batch = collator([short, long])

    assert batch["input_ids"].shape[0] == 1
    assert batch[OVERLENGTH_COUNT_KEY].item() == 1
    assert batch[ORIGINAL_COUNT_KEY].item() == 2
    decoded = processor.tokenizer.decode(batch["labels"][0][batch["labels"][0] != -100])
    assert '"ok":true' in decoded

    empty_batch = collator([long])
    assert "input_ids" not in empty_batch
    assert empty_batch[OVERLENGTH_COUNT_KEY].item() == 1
    assert empty_batch[ORIGINAL_COUNT_KEY].item() == 1


def test_collator_error_policy_names_overlength_source(processor):
    collator = AlignmentCollator(
        processor,
        prompt="Extract JSON.",
        min_pixels=65_536,
        max_pixels=65_536,
        max_sequence_length=512,
        overlength_policy="error",
    )
    with pytest.raises(RuntimeError, match="long-label.png"):
        collator(
            [
                {
                    "image": Image.new("RGB", (64, 96), "white"),
                    "target": json.dumps({"text": "word " * 2_000}),
                    "source": "long-label.png",
                }
            ]
        )


class FakeFullModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.visual = torch.nn.Module()
        self.model.visual.encoder = torch.nn.Linear(3, 4)
        self.model.visual.merger = torch.nn.Linear(4, 5)
        self.language_model = torch.nn.Linear(5, 7)


def test_freezing_and_alignment_delta_export(tmp_path):
    model = FakeFullModel()
    merger, vision = configure_trainable_parameters(model, train_vision=False)
    assert merger
    assert not vision
    assert all(
        parameter.requires_grad for parameter in model.model.visual.merger.parameters()
    )
    assert all(
        not parameter.requires_grad
        for parameter in model.model.visual.encoder.parameters()
    )
    assert all(
        not parameter.requires_grad for parameter in model.language_model.parameters()
    )

    delta = save_alignment_delta(
        model,
        tmp_path,
        base_model="test/chandra",
        mae_delta=tmp_path / "mae.safetensors",
        steps=10,
        train_vision=False,
    )
    assert delta.name == "alignment_merger_delta.safetensors"
    from safetensors import safe_open

    with safe_open(delta, framework="pt") as handle:
        assert handle.keys()
        assert all(key.startswith("model.visual.merger.") for key in handle.keys())
