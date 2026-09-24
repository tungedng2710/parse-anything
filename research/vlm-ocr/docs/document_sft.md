# Benchmarking and Document SFT

After MAE and merger alignment, first compare complete-model generation on a
held-out set. Only start document SFT after the aligned model improves over the
MAE-only model and remains competitive with original Chandra.

Install the project with the Unsloth backend and experiment trackers:

```bash
pip install -e '.[unsloth]'
```

## 1. Benchmark alignment

Copy the example and set `dataset` to the paired dataset root:

```bash
cp docs/benchmark_alignment_example.yaml configs/benchmark_alignment_local.yaml
```

The benchmark loads one Chandra model and evaluates these states sequentially:

1. Original Chandra.
2. Original Chandra plus the MAE vision delta.
3. MAE vision delta plus the alignment merger delta.

Run it on one GPU:

```bash
conda activate main
CUDA_VISIBLE_DEVICES=0 python benchmark_alignment.py \
  --config configs/benchmark_alignment_local.yaml
```

For a quick smoke test, add `--max-samples 4 --max-new-tokens 2048`.

Outputs are written beneath `outputs/chandra2-alignment-benchmark`:

- `base.jsonl`, `mae.jsonl`, and `aligned.jsonl`: per-example predictions.
- `summary.json`: JSON validity, exact match, normalized edit similarity, and
  micro field precision/recall/F1.

Use a held-out dataset that was not used to tune alignment hyperparameters.
Inspect predictions manually in addition to aggregate metrics.

## 2. Train document SFT

SFT uses the same paired layout:

```text
sft_dataset/
├── images/
│   └── example.png
└── labels/
    └── example.json
```

Copy and edit the configuration:

```bash
cp docs/sft_example.yaml configs/sft_local.yaml
```

The default `backend: unsloth` policy loads Chandra through
`FastVisionModel` in 16-bit precision, then applies the MAE vision delta and the
alignment merger delta. It freezes the vision encoder, trains the merger, and
adds Unsloth LoRA to the Qwen 3.5 full-attention, gated-delta, and MLP
projections. Loss is applied only to the assistant JSON response. The model is
kept in 16-bit rather than 4-bit so the learned MAE and merger weights are not
requantized.

Keep `lora_dropout: 0.0`; nonzero LoRA dropout disables part of Unsloth's fast
LoRA path.

Use `backend: transformers` for the standard `AutoModel` loader fallback. If
Unsloth is installed, its import-time patches can still be active in that
process.

Run two steps before a full job:

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch \
  --num_processes 1 \
  --num_machines 1 \
  --mixed_precision bf16 \
  --dynamo_backend no \
  train_sft.py \
  --config configs/sft_local.yaml \
  --max-steps 2 \
  --gradient-accumulation-steps 1 \
  --eval-every 2 \
  --save-every 2
```

Then run the configured schedule:

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch \
  --num_processes 1 \
  --num_machines 1 \
  --mixed_precision bf16 \
  --dynamo_backend no \
  train_sft.py --config configs/sft_local.yaml
```

## Live training dashboards

TensorBoard is enabled by default:

```yaml
report_to: tensorboard
tracker_project_name: chandra-sft
tracker_run_name: chandra-sft-run
```

Start the dashboard in another terminal:

```bash
tensorboard --logdir outputs/chandra2-sft/logs --port 6006
```

Then open `http://localhost:6006`. For a remote machine, forward the port with
SSH, for example `ssh -L 6006:localhost:6006 user@server`.

To use Weights & Biases:

```bash
wandb login
```

```yaml
report_to: wandb
tracker_project_name: chandra-sft
tracker_run_name: h200-sft-run-1
```

Set `report_to: tensorboard,wandb` to log to both, or `report_to: none` to
disable external trackers. Training/validation loss, both learning rates,
elapsed time, and the cumulative overlength-sample count are logged. The local
`metrics.jsonl` file is always retained.

Resume from a full training checkpoint:

```bash
accelerate launch train_sft.py \
  --config configs/sft_local.yaml \
  --resume-from outputs/chandra2-sft/checkpoints/step-250
```

Final outputs include:

- `lora_adapter/`: language-model LoRA adapter.
- `sft_merger_delta.safetensors`: final merger weights.
- `sft_manifest.json`: exact parent checkpoints and loading metadata.
- `processor/`: processor and tokenizer files.

## 3. Load the final SFT model

```python
import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText

from chandra_mae.checkpoint import apply_vision_delta

model = AutoModelForImageTextToText.from_pretrained(
    "datalab-to/chandra-ocr-2",
    dtype=torch.bfloat16,
)
apply_vision_delta(
    model, "outputs/chandra2-mae-full/chandra_vision_delta.safetensors"
)
apply_vision_delta(
    model, "outputs/chandra2-alignment/alignment_merger_delta.safetensors"
)
apply_vision_delta(model, "outputs/chandra2-sft/sft_merger_delta.safetensors")
model = PeftModel.from_pretrained(model, "outputs/chandra2-sft/lora_adapter")
```

For single-image inference, use the included CLI. It reads the manifest and
applies every checkpoint in the required order automatically:

```bash
conda activate main
CUDA_VISIBLE_DEVICES=0 python infer_single.py test_samples/test1.png \
  --sft-dir outputs/chandra2-sft \
  --output outputs/chandra2-sft/test1-inference.json \
  --max-new-tokens 8192 \
  --require-valid-json
```

The command prints and saves the raw model response, parsed JSON, validity,
generation time, token count, and throughput. Omit `--require-valid-json` when
you want to retain invalid or truncated output without a nonzero exit status.

## 4. Train the fair baseline

Use the identical SFT dataset and settings but initialize from original Chandra:

```bash
accelerate launch train_sft.py \
  --config configs/sft_local.yaml \
  --no-load-adapted-vision \
  --output-dir outputs/chandra2-direct-sft
```

The important comparison is aligned-MAE SFT versus direct SFT, not their raw
training losses. Evaluate both using the same generation benchmark and held-out
documents.
