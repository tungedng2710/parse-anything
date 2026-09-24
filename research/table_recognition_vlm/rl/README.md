# Table Recognition SFT + RLVR

Fine-tunes `datalab-to/chandra-ocr-2` with LoRA SFT followed by vision GRPO and
verifiable rewards on `tungedng2710/table_html_with_reasoning`. The policy input
is one or more table images and the policy output is only complete
`<table>...</table>` HTML.

The default instruction lives in [`prompt.md`](prompt.md), rather than being
embedded in the training script. It specifies the transcription, structural
HTML, merged-cell, multi-image, and output-only requirements. Supply a custom
UTF-8 instruction with `--prompt-file /path/to/prompt.md`; an absent or empty
file fails before the dataset or model is loaded.

## Dataset Findings

The Hugging Face dataset has 28,341 train rows and 3,149 test rows. Its columns
include `images`, `table_html`, a `logical_table_reasoning_v1.0` JSON trace,
table dimensions, cell count, merged-cell status, and validation status.
Some labels are full HTML documents; preprocessing extracts only their complete
`<table>...</table>` element before they are passed to the verifier.

The trace is structured privileged supervision, not a desirable model output.
It contains cell coordinates and spans, a logical grid, relations, validation
checks, and a deterministic placement trace. Including it in the prompt would
leak the answer; training the model to emit it would also conflict with the
HTML-only inference contract. This implementation therefore:

- never puts reasoning in the prompt or completion;
- uses `num_rows`, `num_cols`, `num_cells`, and `has_merged_cells`, which are
  compact facts materialized from the trace, as private verifier inputs;
- removes the raw reasoning JSON before training because it is very large;
- filters `validation_passed=False` rows by default (535 train rows); and
- keeps the test split separate for evaluation.

All current rows contain one image, but the prompt and trainer retain list-of-
images support.

## Training Stages

Training is intentionally split into two stages:

1. `sft.py` teaches the base model the image-to-HTML task and exact output
   contract using completion-only supervised loss.
2. `grpo.py` continues the SFT LoRA adapter with verifiable rewards.

Both stages use the same model, prompt, dataset filters, extracted table-only
targets, sequence length, and LoRA target-module policy. SFT conversations are
created lazily so decoded images for the complete dataset are not retained in
memory. The raw reasoning trace is never an SFT target.

## Rewards

`reward.py` contains deterministic rewards with no learned reward model:

| Reward | Weight | Purpose |
| --- | ---: | --- |
| `format_reward` | 0.25 | Exactly one balanced table, no prose, reasoning, or Markdown |
| `exact_reward` | 1.0 | Canonical HTML match while ignoring indentation and attribute order |
| `structure_reward` | 1.0 | Ordered table/section/row/cell tags and rowspan/colspan agreement |
| `content_reward` | 1.0 | Ordered normalized visible-text token agreement by cell |
| `reasoning_metadata_reward` | 0.5 | Rows, logical columns, physical cells, and merged-cell status match trace facts |

The dense structure and content rewards avoid the all-zero groups caused by an
exact-match-only verifier. The format reward enforces the deployment contract,
and exact match supplies a high-confidence terminal signal. Parsing results are
cached and sequence comparisons are linear in table size so reward computation
remains practical for long tables.

## Loss

Training uses TRL's clipped GRPO objective with group-relative advantages,
`epsilon=0.2`, and a small reference-policy KL coefficient `beta=0.001`.
`loss_type="dr_grpo"` uses a constant completion-length normalizer instead of
normalizing each sample by its generated length. This matters here because
table lengths vary substantially and per-sample normalization can bias updates
toward short outputs. Reward standard-deviation scaling is disabled as
recommended for Dr. GRPO, while rewards are still centered within each group.
Truncated completions are masked because a partial table should not become a
positive policy target. `loss.py` provides a small readable reference version;
the actual run uses TRL/Unsloth's optimized implementation.

The default `num_iterations=1` performs one update per sampled group, so the
old and current policies initially coincide and clipping is normally inactive.
Set `--num-iterations 2` or higher to reuse each sampled group for multiple
updates where the clipping bound becomes active, at additional compute cost.

## Install

Activate the requested environment and install any missing dependencies:

```bash
conda activate main
pip install -U unsloth unsloth_zoo datasets trl accelerate pillow torch
```

The implementation was checked against Unsloth `2026.6.9`, TRL `0.24.0`,
Transformers `5.13.1`, and Datasets `4.3.0`.
`grpo.py` includes a narrow, architecture-gated generation-signature workaround
for this Unsloth version. It is applied only when the loaded model reports a
Qwen3.5 base architecture whose compiled wrapper would otherwise reject
`mm_token_type_ids` before the first multimodal generation.
The installed FLA gradient kernel also requests more shared memory than an H200
can provide for this architecture. This script therefore uses Transformers'
Torch gated-delta fallback for both generation and training.

## Validate

Run unit tests and a dataset dry run before allocating the model:

```bash
conda activate main
python -m unittest -v test_reward.py test_loss.py
python -m unittest -v test_grpo.py test_sft.py
python sft.py --dry-run --max-train-samples 16 --max-eval-samples 8
python grpo.py --dry-run --max-train-samples 16 --max-eval-samples 8
```

## Stage 1: LoRA SFT

The default SFT run uses 4-bit QLoRA, completion-only loss, a batch size of one,
eight accumulation steps, and one epoch:

```bash
conda activate main
python sft.py
```

The resulting adapter and processor are saved to
`chandra_ocr_2_table_html_sft`. Resume an interrupted SFT trainer checkpoint
with:

```bash
python sft.py \
  --resume-from-checkpoint chandra_ocr_2_table_html_sft/checkpoint-1000
```

## Stage 2: GRPO

Start GRPO from the saved SFT adapter by passing its directory as the model.
The loader detects its PEFT configuration and continues that adapter instead of
creating a second one:

```bash
conda activate main
python grpo.py \
  --model-name chandra_ocr_2_table_html_sft \
  --output-dir chandra_ocr_2_table_html_sft_grpo
```

Do not use `--resume-from-checkpoint` to move from SFT to GRPO; that option is
only for resuming a checkpoint produced by the same trainer stage. When GRPO
loads an existing adapter, its LoRA rank and target modules come from the SFT
adapter, so GRPO's `--lora-*` and `--language-only` creation options do not
replace them. For a local adapter, GRPO reads `adapter_config.json` and
automatically expands Unsloth's rank allocation when SFT used a larger rank.

To use 16-bit LoRA or language-only adapters, choose those options during SFT
and use the same quantization mode for GRPO:

```bash
python sft.py --load-in-16bit
python grpo.py --model-name chandra_ocr_2_table_html_sft --load-in-16bit

python sft.py --language-only
python grpo.py --model-name chandra_ocr_2_table_html_sft
```

Resume an interrupted GRPO trainer checkpoint:

```bash
python grpo.py \
  --model-name chandra_ocr_2_table_html_sft \
  --output-dir chandra_ocr_2_table_html_sft_grpo \
  --resume-from-checkpoint chandra_ocr_2_table_html_sft_grpo/checkpoint-100
```

Both the effective training generation batch and global evaluation batch must
be divisible by `num_generations`. Long references can exceed the default
8,192-token generation cap; truncated outputs are intentionally excluded from
the policy loss. Increase both `--max-completion-length` and, if necessary,
`--max-seq-length` when memory permits. For a cold base model that produces no
valid tables across a generation group, run supervised table-HTML fine-tuning
before RLVR. Colocated vLLM is intentionally disabled because the checked
Unsloth release does not support this Qwen3.5 vision training path reliably.
