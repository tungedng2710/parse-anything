"""Completion-only multimodal SFT with frozen vision and language-model LoRA."""
from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path
import re

from .common import dump_json, read_manifest, verify_disjoint
from .modeling import full_text, load_base, load_images, load_processor, prompt_text


class ExtractionCollator:
    """Keep image tokens intact and supervise JSON completion plus its terminator.

    Use the native NuExtract chat template for both training and inference. The
    tokenizer expands vision placeholders, so compute prompt lengths with the
    processor, not a text-only tokenizer. Fail if the prefix does not align.
    """

    def __init__(self, processor, input_mode: str, max_length: int):
        self.processor, self.input_mode, self.max_length = processor, input_mode, max_length

    def __call__(self, rows: list[dict]) -> dict:
        images = [load_images(row) for row in rows]
        texts = [full_text(self.processor, row, imgs, self.input_mode) for row, imgs in zip(rows, images)]
        batch = self.processor(text=texts, images=images, padding=True, truncation=False,
                               add_special_tokens=False, return_tensors="pt")
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        for i, (row, imgs) in enumerate(zip(rows, images)):
            prefix = self.processor(text=[prompt_text(self.processor, row, imgs, self.input_mode)],
                                    images=imgs, padding=False, truncation=False,
                                    add_special_tokens=False, return_tensors="pt")["input_ids"][0]
            length = int(batch["attention_mask"][i].sum())
            if length > self.max_length:
                raise ValueError(f"{row['id']}: {length} tokens exceeds {self.max_length}; "
                                 "increase the guard or reduce image pixels/schema size. Never truncate images.")
            if len(prefix) >= length or not batch["input_ids"][i, :len(prefix)].equal(prefix):
                raise ValueError(f"{row['id']}: native prompt/completion boundary does not align")
            labels[i, :len(prefix)] = -100
        batch["labels"] = labels
        return batch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("artifacts/tt32"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/nuextract3-tt32-lora"))
    parser.add_argument("--model", default="numind/NuExtract3")
    parser.add_argument("--revision", default="main", help="Use an immutable HF commit for reproducible runs")
    parser.add_argument("--input-mode", choices=["image", "image-text"], default="image")
    parser.add_argument("--qlora", action="store_true")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--max-pixels", type=int, default=1605632,
                        help="Per page pixel budget; reduce for memory, ablate for dense grids")
    parser.add_argument("--max-length", type=int, default=32768, help="Reject oversized sequences; never truncate")
    parser.add_argument("--attention", choices=["sdpa", "eager", "flash_attention_2"], default="sdpa")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    for name in ("epochs", "learning_rate", "rank", "batch_size", "gradient_accumulation", "max_pixels", "max_length"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        parser.error("Output is nonempty; choose a fresh output or --resume")
    if args.resume and not args.resume.is_dir():
        parser.error("--resume must point to an existing Trainer checkpoint")
    train_rows = read_manifest(args.data / "train.jsonl")
    val_rows = read_manifest(args.data / "validation.jsonl")
    test_rows = read_manifest(args.data / "test.jsonl")
    verify_disjoint(train_rows, val_rows, test_rows)
    if not train_rows or not val_rows:
        parser.error("Train and validation must both contain documents")

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required. Training has not been started.")
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import Trainer, TrainingArguments, set_seed
    set_seed(args.seed)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    processor = load_processor(args.model, args.revision, args.max_pixels)
    model = load_base(args.model, args.revision, dtype, args.qlora, args.attention)
    model.config.use_cache = False
    if args.qlora:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                                gradient_checkpointing_kwargs={"use_reentrant": False})
    # Match both Qwen3.5 full and linear attention plus MLP; never vision/lm_head.
    suffixes = {"q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z",
                "in_proj_b", "in_proj_a", "out_proj", "gate_proj", "up_proj", "down_proj"}
    targets = [name for name, _ in model.named_modules()
               if "language_model" in name and name.rsplit(".", 1)[-1] in suffixes]
    if not targets:
        raise RuntimeError("No Qwen3.5 language LoRA targets found; inspect the checkpoint architecture")
    target_pattern = "(?:" + "|".join(re.escape(name) for name in targets) + ")"
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=args.rank * 2,
                           lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                           target_modules=target_pattern))
    if any(p.requires_grad and ("visual" in name or "lm_head" in name)
           for name, p in model.named_parameters()):
        raise RuntimeError("Vision tower or output head unexpectedly trainable")
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    training_args = TrainingArguments(
        output_dir=str(args.output), num_train_epochs=args.epochs, learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=dtype == torch.bfloat16, fp16=dtype == torch.float16,
        eval_strategy="epoch", save_strategy="epoch", logging_steps=1, save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        warmup_ratio=0.05, lr_scheduler_type="cosine", weight_decay=0.01,
        remove_unused_columns=False, report_to="none", seed=args.seed, data_seed=args.seed,
        dataloader_num_workers=0, ddp_find_unused_parameters=False, prediction_loss_only=True)
    trainer = Trainer(model=model, args=training_args, train_dataset=train_rows,
                      eval_dataset=val_rows, processing_class=processor,
                      data_collator=ExtractionCollator(processor, args.input_mode, args.max_length))
    # PeftModel forwards **kwargs; the standard causal loss does not need Trainer's
    # extra num_items_in_batch keyword and uses normal accumulation scaling.
    trainer.model_accepts_loss_kwargs = False
    trainer.train(resume_from_checkpoint=str(args.resume) if args.resume else None)
    trainer.save_model(str(args.output / "adapter"))
    if trainer.is_world_process_zero():
        processor.save_pretrained(args.output / "adapter")
        config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        config.update({"resolved_model_revision": getattr(model.config, "_commit_hash", None),
                       "packages": {name: importlib.metadata.version(name) for name in
                                    ("torch", "transformers", "peft", "accelerate")},
                       "lora_target_modules": targets, "best_checkpoint": trainer.state.best_model_checkpoint,
                       "train_documents": len(train_rows), "validation_documents": len(val_rows)})
        dump_json(args.output / "run_config.json", config)


if __name__ == "__main__":
    main()
