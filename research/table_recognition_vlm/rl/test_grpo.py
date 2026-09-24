import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from grpo import load_prompt, max_lora_rank_for_model, patch_qwen35_generation_inputs


class _Config:
    model_type = "qwen3_5"


class _Model:
    config = _Config()

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **kwargs):
        return {"input_ids": input_ids, "attention_mask": attention_mask, **kwargs}


class GenerationPatchTest(unittest.TestCase):
    def test_patch_exposes_and_forwards_mm_token_type_ids(self):
        model = _Model()
        self.assertTrue(patch_qwen35_generation_inputs(model))
        values = model.prepare_inputs_for_generation(
            [1, 2], attention_mask=[1, 1], mm_token_type_ids=[0, 1]
        )
        self.assertEqual(values["mm_token_type_ids"], [0, 1])
        self.assertFalse(patch_qwen35_generation_inputs(model))


class PromptTest(unittest.TestCase):
    def test_load_prompt_strips_surrounding_whitespace(self):
        with TemporaryDirectory() as directory:
            path = Path(directory, "prompt.md")
            path.write_text("\n  Convert the table.  \n", encoding="utf-8")
            self.assertEqual(load_prompt(path), "Convert the table.")

    def test_load_prompt_rejects_empty_file(self):
        with TemporaryDirectory() as directory:
            path = Path(directory, "prompt.md")
            path.write_text(" \n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_prompt(path)


class AdapterConfigTest(unittest.TestCase):
    def test_local_adapter_rank_expands_unsloth_ceiling(self):
        with TemporaryDirectory() as directory:
            path = Path(directory, "adapter_config.json")
            path.write_text(
                json.dumps({"r": 32, "rank_pattern": {"vision": 64}}),
                encoding="utf-8",
            )
            self.assertEqual(max_lora_rank_for_model(directory, 16), 64)

    def test_base_model_uses_requested_rank(self):
        with TemporaryDirectory() as directory:
            self.assertEqual(max_lora_rank_for_model(directory, 16), 16)


if __name__ == "__main__":
    unittest.main()
