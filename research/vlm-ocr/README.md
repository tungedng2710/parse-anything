# Chandra 2 document MAE training

This repository implements the first experimental branch in the
[Chandra 2 training plan](docs/chandra2_document_ai_training_plan.md):
Chandra's native vision tower is adapted with 75% random masking and masked-only
MSE against resized raw RGB patches. The original multimodal merger and language
model are not involved.

The implementation is an asymmetric MAE: only the 25% visible tokens pass
through the Chandra encoder. A temporary 4-layer decoder restores masked token
positions and is discarded after training. It uses the model's native image
processor, patch embedding, learned position interpolation, rotary positions,
and transformer blocks.

## Environment and sample data

Run commands through the requested environment:

```bash
conda activate main
```

The checked-out test dataset contains a uniform random sample copied from the
source corpus. To reproduce it into an empty directory with a fixed seed:

```bash
python sample_images.py /media/drive-2t/main/idp/data/images sample_dataset \
  --count 10000 --seed 42
```

## Train

The first run downloads the Chandra checkpoint. Chandra currently publishes a
single safetensors file, but only vision tensors are materialized in CPU memory.
Launch on one or more GPUs with Accelerate:

```bash
cp configs/example.yaml configs/local.yaml
accelerate launch train_mae.py --config configs/local.yaml
```

Edit `configs/local.yaml` for the local dataset, output directory, and training
schedule. Configuration YAML files are ignored except for the checked-in
`configs/example.yaml` template.

The progress bar reports global optimizer steps and the current gradient
accumulation microbatch (for example, `accumulation=137/512`) so long effective
batches remain visibly active before the next JSON metric is written.

For a short integration run, override the schedule and decoder size:

```bash
accelerate launch train_mae.py --config configs/local.yaml \
  --max-steps 2 --gradient-accumulation-steps 1 \
  --decoder-hidden-size 128 --decoder-layers 1 --decoder-heads 4
```

Important outputs are:

- `vision_encoder/`: standalone adapted Chandra vision tower.
- `chandra_vision_delta.safetensors`: weights with full-model key prefixes.
- `vision_delta_manifest.json`: base model and compatibility metadata.
- `mae_decoder.pt`: temporary decoder, retained only for diagnostics/resume.
- `checkpoints/step-N/`: full Accelerate training state.
- `metrics.jsonl`: reconstruction loss and learning rates.

To reconnect the encoder for alignment/SFT, load the full Chandra model and
call `chandra_mae.checkpoint.apply_vision_delta(model, delta_path)`. The delta
uses exact original parameter names, so unexpected keys are rejected. Perform
the short projector/merger alignment stage described in the plan before final
document SFT; those supervised stages require paired OCR/Markdown/layout labels
and are deliberately separate from this raw-image pipeline.

See the [vision-language alignment guide](docs/vision_language_alignment.md)
for the required paired-data format, model freezing policy, assistant-only
loss, recommended hyperparameters, checkpointing, and evaluation procedure.

For a paired `images/` and `labels/` dataset, copy the alignment configuration
and launch the supervised merger-alignment trainer:

```bash
cp docs/alignment_example.yaml configs/alignment_local.yaml
accelerate launch train_alignment.py --config configs/alignment_local.yaml
```

After alignment, benchmark base/MAE/aligned generation and train document SFT:

```bash
cp docs/benchmark_alignment_example.yaml configs/benchmark_alignment_local.yaml
python benchmark_alignment.py --config configs/benchmark_alignment_local.yaml

cp docs/sft_example.yaml configs/sft_local.yaml
accelerate launch train_sft.py --config configs/sft_local.yaml
```

See the [benchmark and document SFT guide](docs/document_sft.md) for smoke tests,
checkpoint loading, resume, and the required direct-SFT baseline.

SFT uses Unsloth's 16-bit `FastVisionModel` backend by default. TensorBoard is
enabled for alignment and SFT; run `tensorboard --logdir outputs/chandra2-sft/logs`
to view the live SFT dashboard. W&B can be selected with `report_to: wandb`.

## View the masks

Render deterministic side-by-side previews using the same preprocessing and
patch order as training:

```bash
python visualize_masks.py --count 4 --seed 42 --output previews/masked_samples.png
```

## Validate

```bash
pytest -q
python -m compileall -q chandra_alignment chandra_mae benchmark_alignment.py train_alignment.py train_mae.py train_sft.py
```
