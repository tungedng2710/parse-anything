from __future__ import annotations

import json

import torch
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import AutoModel, Qwen3_5VisionConfig

from chandra_mae.checkpoint import load_chandra_vision, save_vision_delta
from chandra_mae.data import StreamingDocumentImageDataset, unpatchify_chandra_image
from chandra_mae.model import ChandraMAE


def tiny_vision():
    config = Qwen3_5VisionConfig(
        depth=2,
        hidden_size=64,
        intermediate_size=128,
        num_heads=4,
        in_channels=3,
        patch_size=4,
        temporal_patch_size=2,
        spatial_merge_size=2,
        num_position_embeddings=64,
        out_hidden_size=64,
    )
    return AutoModel.from_config(config)


def test_mae_masks_raw_patches_and_backpropagates():
    torch.manual_seed(7)
    model = ChandraMAE(
        tiny_vision(),
        decoder_hidden_size=32,
        decoder_layers=1,
        decoder_heads=4,
        mask_ratio=0.75,
    )
    # Two 4x4 patch grids. Qwen patches flatten C*T*P*P values.
    grid = torch.tensor([[1, 4, 4], [1, 4, 4]])
    pixels = torch.randn(32, 3 * 2 * 4 * 4)
    targets = torch.rand_like(pixels)
    output = model(pixels, grid, targets)
    assert output.predictions.shape == targets.shape
    assert output.mask.shape == (32,)
    assert output.mask[:16].sum() == 12
    assert output.mask[16:].sum() == 12
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert model.vision.blocks[0].attn.qkv.weight.grad is not None
    assert model.pixel_head.weight.grad is not None
    assert all(parameter.grad is None for parameter in model.vision.merger.parameters())


def test_mae_bfloat16_autocast():
    model = ChandraMAE(
        tiny_vision(),
        decoder_hidden_size=32,
        decoder_layers=1,
        decoder_heads=4,
        mask_ratio=0.75,
    )
    grid = torch.tensor([[1, 4, 4]])
    pixels = torch.randn(16, 3 * 2 * 4 * 4)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = model(pixels, grid, torch.rand_like(pixels))
    assert torch.isfinite(output.loss)
    output.loss.backward()


def test_streaming_dataset_and_prefixed_delta(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for index in range(6):
        Image.new("RGB", (8, 8), (index, index, index)).save(image_dir / f"{index}.png")
    dataset = StreamingDocumentImageDataset(
        image_dir, recursive=False, seed=3, shuffle_buffer=2
    )
    examples = list(dataset)
    assert len(examples) == 6
    assert len({example["path"] for example in examples}) == 6

    rank_zero = StreamingDocumentImageDataset(
        image_dir,
        recursive=False,
        seed=3,
        shuffle_buffer=2,
        process_index=0,
        num_processes=2,
    )
    rank_one = StreamingDocumentImageDataset(
        image_dir,
        recursive=False,
        seed=3,
        shuffle_buffer=2,
        process_index=1,
        num_processes=2,
    )
    zero_paths = {example["path"] for example in rank_zero}
    one_paths = {example["path"] for example in rank_one}
    assert zero_paths.isdisjoint(one_paths)
    assert zero_paths | one_paths == {example["path"] for example in examples}

    output_dir = tmp_path / "checkpoint"
    delta = save_vision_delta(
        tiny_vision(),
        output_dir,
        base_model="test/chandra",
        base_prefix="model.visual.",
    )
    with safe_open(delta, framework="pt") as handle:
        keys = list(handle.keys())
        assert keys
        assert all(key.startswith("model.visual.") for key in keys)
    assert (output_dir / "vision_encoder" / "model.safetensors").is_file()
    reloaded, prefix = load_chandra_vision(
        str(output_dir / "vision_encoder"), local_files_only=True
    )
    assert prefix == ""
    assert (
        reloaded.patch_embed.proj.weight.shape
        == tiny_vision().patch_embed.proj.weight.shape
    )


def test_loads_sharded_vision_checkpoint_without_unrelated_shards(tmp_path):
    checkpoint_dir = tmp_path / "sharded"
    tiny_vision().save_pretrained(checkpoint_dir, safe_serialization=True)
    state = load_file(checkpoint_dir / "model.safetensors")
    names = sorted(state)
    midpoint = len(names) // 2
    shard_names = [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]
    weight_map = {}
    for shard_name, shard_keys in zip(
        shard_names, (names[:midpoint], names[midpoint:]), strict=True
    ):
        prefixed = {f"model.visual.{name}": state[name] for name in shard_keys}
        save_file(prefixed, checkpoint_dir / shard_name)
        weight_map.update({name: shard_name for name in prefixed})
    # The loader must not require a shard that contains only language weights.
    weight_map["model.language_model.layers.0.fake.weight"] = (
        "model-00003-of-00003.safetensors"
    )
    (checkpoint_dir / "model.safetensors").unlink()
    (checkpoint_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map})
    )

    reloaded, prefix = load_chandra_vision(
        str(checkpoint_dir), local_files_only=True
    )
    assert prefix == "model.visual."
    assert torch.equal(
        reloaded.patch_embed.proj.weight, state["patch_embed.proj.weight"]
    )


def test_unpatchify_matches_chandra_spatial_merge_order():
    channels, grid_h, grid_w, merge, patch, temporal = 3, 4, 6, 2, 2, 2
    image = torch.arange(channels * grid_h * patch * grid_w * patch).reshape(
        channels, grid_h * patch, grid_w * patch
    )
    patches = (
        image.reshape(
            channels,
            grid_h // merge,
            merge,
            patch,
            grid_w // merge,
            merge,
            patch,
        )
        .permute(1, 4, 2, 5, 0, 3, 6)
        .unsqueeze(5)
        .expand(-1, -1, -1, -1, -1, temporal, -1, -1)
        .reshape(grid_h * grid_w, channels * temporal * patch * patch)
    )
    restored = unpatchify_chandra_image(
        patches,
        torch.tensor([1, grid_h, grid_w]),
        patch_size=patch,
        temporal_patch_size=temporal,
        spatial_merge_size=merge,
        num_channels=channels,
    )
    assert restored.shape == (temporal, *image.shape)
    assert torch.equal(restored[0], image)
    assert torch.equal(restored[1], image)
