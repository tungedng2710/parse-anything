"""Shared, lazily imported GPU model loading and native prompt construction."""
from __future__ import annotations

from .common import template_kwargs, user_messages


def load_images(row: dict) -> list:
    from PIL import Image
    images = []
    for path in row["images"]:
        with Image.open(path) as source:
            images.append(source.convert("RGB"))
    return images


def load_processor(model: str, revision: str, max_pixels: int):
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model, revision=revision)
    processor.tokenizer.padding_side = "right"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    # Qwen3VLProcessor uses Qwen2VLImageProcessor; current v5 uses size bounds.
    processor.image_processor.size["longest_edge"] = max_pixels
    return processor


def load_base(model: str, revision: str, dtype, qlora: bool, attention: str):
    import os
    import torch
    from transformers import AutoModelForImageTextToText, BitsAndBytesConfig
    if not torch.cuda.is_available():
        raise RuntimeError("This pipeline requires a CUDA GPU; no model was downloaded or trained.")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    kwargs = {"revision": revision, "dtype": dtype, "attn_implementation": attention}
    if qlora:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
        kwargs["device_map"] = {"": local_rank}
    return AutoModelForImageTextToText.from_pretrained(model, **kwargs)


def prompt_text(processor, row: dict, images: list, input_mode: str,
                include_instructions: bool = True) -> str:
    kwargs = template_kwargs(row)
    if not include_instructions:
        kwargs.pop("instructions")
    return processor.apply_chat_template(
        user_messages(row, images, input_mode), tokenize=False,
        add_generation_prompt=True, **kwargs)


def full_text(processor, row: dict, images: list, input_mode: str) -> str:
    import json
    messages = user_messages(row, images, input_mode)
    messages.append({"role": "assistant", "content": json.dumps(row["label"], ensure_ascii=False)})
    return processor.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=False, **template_kwargs(row))
