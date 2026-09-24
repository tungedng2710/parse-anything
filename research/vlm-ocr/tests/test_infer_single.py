from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image

from infer_single import generate_single, resolve_sft_artifacts


def test_resolve_sft_artifacts_from_manifest(tmp_path: Path, monkeypatch):
    run = tmp_path / "sft"
    run.mkdir()
    mae = tmp_path / "mae.safetensors"
    alignment = tmp_path / "alignment.safetensors"
    mae.touch()
    alignment.touch()
    (run / "sft_merger_delta.safetensors").touch()
    (run / "lora_adapter").mkdir()
    (run / "processor").mkdir()
    (run / "sft_manifest.json").write_text(
        json.dumps(
            {
                "base_model": "test/chandra",
                "mae_delta": str(mae),
                "alignment_delta": str(alignment),
                "merger_delta": "sft_merger_delta.safetensors",
                "lora_adapter": "lora_adapter",
            }
        )
    )

    artifacts = resolve_sft_artifacts(run)
    assert artifacts.base_model == "test/chandra"
    assert artifacts.mae_delta == mae
    assert artifacts.alignment_delta == alignment
    assert artifacts.merger_delta == run / "sft_merger_delta.safetensors"
    assert artifacts.lora_adapter == run / "lora_adapter"
    assert artifacts.processor == run / "processor"


class FakeProcessor:
    tokenizer = type("Tokenizer", (), {"eos_token_id": 9, "pad_token_id": 8})()

    def apply_chat_template(self, *args, **kwargs):
        return "prompt"

    def __call__(self, **kwargs):
        return {
            "input_ids": torch.ones(1, 3, dtype=torch.long),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
        }

    def batch_decode(self, values, **kwargs):
        assert values.shape == (1, 2)
        return ['{"ok":true}']


class FakeModel(torch.nn.Module):
    generation_config = type("GenerationConfig", (), {"eos_token_id": 7})()

    def generate(self, input_ids, attention_mask, **kwargs):
        assert kwargs["eos_token_id"] == [7, 9]
        assert kwargs["pad_token_id"] == 8
        continuation = torch.ones(1, 2, dtype=torch.long)
        return torch.cat((input_ids, continuation), dim=1)


def test_generate_single_decodes_only_continuation():
    response, token_count, elapsed = generate_single(
        FakeModel(),
        FakeProcessor(),
        Image.new("RGB", (10, 10)),
        "extract",
        device=torch.device("cpu"),
        dtype=torch.float32,
        min_pixels=16,
        max_pixels=100,
        max_input_length=32,
        max_new_tokens=8,
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
    )
    assert response == '{"ok":true}'
    assert token_count == 2
    assert elapsed >= 0
