from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from benchmark_alignment import (
    BenchmarkConfig,
    batches,
    evaluate_variant,
    resolve_artifacts,
)
from chandra_alignment.tracking import tracker_names
from chandra_alignment.metrics import (
    aggregate_scores,
    extract_json,
    score_json_prediction,
)
from train_sft import (
    SFTConfig,
    configure_sft_parameters,
    resolve_sft_inputs,
    save_sft_artifacts,
)


def test_extract_and_score_json_response():
    prediction = '<think>checking</think>\n```json\n{"b":2,"a":1}\n```'
    assert extract_json(prediction) == {"a": 1, "b": 2}
    score = score_json_prediction(prediction, {"a": 1, "b": 2})
    assert score["json_valid"]
    assert score["exact_match"]
    assert score["edit_similarity"] == 1.0
    assert score["field_true_positives"] == 2


@pytest.mark.parametrize(
    "response",
    [
        '{"ok":true} trailing text',
        '{"ok":true}\n{"second":true}',
        'broken {"nested":{"looks_valid":true}} trailing',
        'assistant\n<think>again</think>\n{"ok":true} garbage',
    ],
)
def test_extract_json_rejects_trailing_or_nested_fragments(response):
    assert extract_json(response) is None


def test_aggregate_scores_uses_micro_field_f1():
    scores = [
        score_json_prediction('{"a":1,"b":2}', {"a": 1, "b": 3}),
        score_json_prediction("not json", {"c": 4}),
    ]
    summary = aggregate_scores(scores)
    assert summary["samples"] == 2
    assert summary["json_valid_rate"] == 0.5
    assert summary["exact_match_rate"] == 0.0
    assert summary["field_precision"] == 0.5
    assert summary["field_recall"] == pytest.approx(1 / 3)
    assert summary["field_f1"] == pytest.approx(0.4)


def test_batches_obeys_limit():
    examples = iter({"value": index} for index in range(10))
    result = list(batches(examples, batch_size=2, limit=5))
    assert [len(batch) for batch in result] == [2, 2, 1]
    assert [item["value"] for batch in result for item in batch] == list(range(5))


def test_tracker_selection():
    assert tracker_names("tensorboard,wandb,tensorboard") == [
        "tensorboard",
        "wandb",
    ]
    assert tracker_names("none") == []
    with pytest.raises(ValueError, match="Unsupported tracker"):
        tracker_names("unknown")


class FakeGenerationProcessor:
    def apply_chat_template(self, *args, **kwargs):
        return "prompt"

    def __call__(self, images, text, **kwargs):
        count = len(images)
        return {
            "input_ids": torch.ones(count, 3, dtype=torch.long),
            "attention_mask": torch.ones(count, 3, dtype=torch.long),
        }

    def batch_decode(self, generated, **kwargs):
        return ['{"ok":true}'] * generated.shape[0]


class FakeGenerationModel(torch.nn.Module):
    def generate(self, input_ids, attention_mask, **kwargs):
        continuation = torch.ones(input_ids.shape[0], 2, dtype=torch.long)
        return torch.cat((input_ids, continuation), dim=1)


def test_benchmark_generation_writes_predictions(tmp_path, monkeypatch):
    examples = [
        {"image": object(), "target": '{"ok":true}', "source": "one.png"},
        {"image": object(), "target": '{"ok":true}', "source": "two.png"},
    ]
    monkeypatch.setattr(
        "benchmark_alignment.make_dataset", lambda config: iter(examples)
    )
    config = BenchmarkConfig(
        dataset="unused",
        output_dir=str(tmp_path),
        max_samples=2,
        batch_size=2,
        mixed_precision="no",
    )
    summary = evaluate_variant(
        "fake",
        FakeGenerationModel(),
        FakeGenerationProcessor(),
        config,
        torch.device("cpu"),
    )
    assert summary["samples"] == 2
    assert summary["json_valid_rate"] == 1.0
    assert summary["exact_match_rate"] == 1.0
    assert len((tmp_path / "fake.jsonl").read_text().splitlines()) == 2


def write_alignment_artifacts(root: Path) -> tuple[Path, Path]:
    root.mkdir()
    mae = root / "mae.safetensors"
    alignment = root / "alignment_merger_delta.safetensors"
    mae.touch()
    alignment.touch()
    (root / "alignment_delta_manifest.json").write_text(
        json.dumps(
            {
                "base_model": "test/chandra",
                "mae_delta": str(mae),
            }
        )
    )
    (root / "alignment_config.json").write_text(
        json.dumps({"dataset": str(root / "dataset")})
    )
    return mae, alignment


def test_benchmark_and_sft_artifact_resolution(tmp_path):
    alignment_dir = tmp_path / "alignment"
    mae, alignment = write_alignment_artifacts(alignment_dir)

    benchmark = BenchmarkConfig(alignment_dir=str(alignment_dir))
    assert resolve_artifacts(benchmark) == ("test/chandra", mae, alignment)
    assert benchmark.dataset == str(alignment_dir / "dataset")

    sft = SFTConfig(alignment_dir=str(alignment_dir))
    assert resolve_sft_inputs(sft)[:3] == ("test/chandra", mae, alignment)

    baseline = SFTConfig(
        base_model="test/chandra",
        alignment_dir=str(alignment_dir),
        load_adapted_vision=False,
    )
    assert resolve_sft_inputs(baseline)[:3] == ("test/chandra", None, None)


class FakeFullModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.visual = torch.nn.Module()
        self.model.visual.encoder = torch.nn.Linear(3, 4)
        self.model.visual.merger = torch.nn.Linear(4, 5)
        self.language_model = torch.nn.Linear(5, 7)


class FakePeftModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = FakeFullModel().requires_grad_(False)
        self.lora_A = torch.nn.Parameter(torch.ones(2, 2))
        self.lora_B = torch.nn.Parameter(torch.ones(2, 2))

    def get_base_model(self):
        return self.base

    def save_pretrained(self, output_dir, safe_serialization=True):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True)
        (output_dir / "adapter_config.json").write_text("{}")


class FakeProcessor:
    def save_pretrained(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True)
        (output_dir / "processor_config.json").write_text("{}")


def test_sft_parameter_policy_trains_only_lora_and_merger():
    model = FakePeftModel()
    lora, merger = configure_sft_parameters(model)
    assert len(lora) == 2
    assert merger
    assert all(parameter.requires_grad for parameter in lora + merger)
    assert all(
        not parameter.requires_grad
        for parameter in model.base.model.visual.encoder.parameters()
    )
    expected = {id(parameter) for parameter in lora + merger}
    assert {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    } == expected


def test_sft_artifact_export(tmp_path):
    model = FakePeftModel()
    configure_sft_parameters(model)
    config = SFTConfig(base_model="test/chandra", lora_rank=4)
    save_sft_artifacts(
        model,
        FakeProcessor(),
        tmp_path,
        config,
        base_model="test/chandra",
        mae_delta=Path("mae.safetensors"),
        alignment_delta=Path("alignment.safetensors"),
        completed_steps=12,
    )
    manifest = json.loads((tmp_path / "sft_manifest.json").read_text())
    assert manifest["steps"] == 12
    assert manifest["lora_rank"] == 4
    assert (tmp_path / "lora_adapter" / "adapter_config.json").is_file()
    assert (tmp_path / "processor" / "processor_config.json").is_file()
    from safetensors import safe_open

    with safe_open(tmp_path / "sft_merger_delta.safetensors", framework="pt") as file:
        assert file.keys()
        assert all(key.startswith("model.visual.merger.") for key in file.keys())
