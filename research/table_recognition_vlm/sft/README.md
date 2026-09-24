# Table Recognition SFT

Fine-tunes `Qwen/Qwen3.5-2B` with Unsloth QLoRA on
`tungedng2710/table_html_with_reasoning`. The reasoning trace in each label is
discarded; only the complete `<table>...</table>` element is supervised.

## Install

```bash
pip install -U unsloth unsloth_zoo datasets trl accelerate pillow
```

## Train

```bash
python train.py
```

Run a small validation before starting GPU training:

```bash
python train.py --dry-run --max-train-samples 16 --max-eval-samples 8
```

Use 16-bit LoRA instead of the default 4-bit QLoRA:

```bash
python train.py --load-in-16bit
```

Resume a run:

```bash
python train.py --resume-from-checkpoint qwen35_2b_table_html_lora/checkpoint-100
```

Evaluation and checkpointing happen at each epoch. `eval_loss` selects the best
checkpoint, and `save_total_limit=2` retains only the best and latest resumable
`checkpoint-*` directories. After training, the output directory root contains
the best LoRA adapter and processor for inference.
