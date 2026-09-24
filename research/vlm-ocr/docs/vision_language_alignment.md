# Chandra 2 Vision-Language Alignment Guide

## Purpose

MAE training changes Chandra's vision features but does not train the
multimodal merger or language model. The next stage teaches the original
Chandra merger how to interpret the adapted features:

```text
image
  -> MAE-adapted vision encoder
  -> trainable Chandra merger
  -> frozen Chandra language model
  -> assistant-response loss
```

This is a supervised stage. Raw document images alone are insufficient;
alignment requires an image, an instruction, and a target response.

The repository provides `train_alignment.py`, which implements the data format,
model setup, assistant-only loss, validation, checkpoint resumption, and compact
delta export described below.

## Prerequisites

Before alignment, confirm that the MAE output contains:

```text
outputs/<mae-run>/
├── chandra_vision_delta.safetensors
├── vision_delta_manifest.json
└── vision_encoder/
```

Use `chandra_vision_delta.safetensors` for alignment. It has parameter names
that match the full Chandra checkpoint. The standalone `vision_encoder/`
directory is useful for vision-only inspection but is not a full VLM.

Run commands in the project environment:

```bash
conda activate main
cd /root/main/cognidoc/research/vlm-ocr
```

## Alignment Dataset

### JSONL format

Store one example per line:

```json
{"image":"/data/pages/001.png","prompt":"Convert this page to Markdown.","target":"# Quarterly report\n\n...","document_id":"report-001","task":"markdown"}
{"image":"/data/pages/002.png","prompt":"Extract all visible text.","target":"CÔNG TY CỔ PHẦN ...","document_id":"report-002","task":"ocr"}
{"image":"/data/pages/003.png","prompt":"Convert the table to HTML.","target":"<table>...</table>","document_id":"report-003","task":"table_html"}
```

Required fields:

- `image`: readable local image path.
- `prompt`: user instruction.
- `target`: assistant response used for language-model loss.

Recommended fields:

- `document_id`: source PDF/document identifier.
- `task`: `ocr`, `markdown`, `table_html`, `layout_json`, and so on.
- `language` and `source`: useful for sampling and evaluation slices.

Prefer native PDF text and layout when available. OCR pseudo-labels are usable,
but filter low-confidence or malformed samples because the frozen language
model will learn how the merger maps visual features to those targets.

### Data splitting

Split by `document_id`, not page path. Pages from the same PDF must not be
distributed between training and validation sets. A reasonable initial split
is 98% training and 2% validation, with at least several thousand validation
pages covering every task and language.

### Task mixture

Start with targets that strongly connect visual content to text:

1. Plain OCR text.
2. Markdown with reading order.
3. Table HTML.
4. Layout or key-value JSON.

Do not begin with document question answering alone. QA targets supervise only
a small subset of page content and provide a weaker projector-alignment signal.

## Load the Full Model and Apply MAE Weights

Load the original full Chandra model first, then apply the MAE delta:

```python
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from chandra_mae.checkpoint import apply_vision_delta

BASE_MODEL = "datalab-to/chandra-ocr-2"
MAE_DELTA = "outputs/chandra2-mae-full/chandra_vision_delta.safetensors"

processor = AutoProcessor.from_pretrained(BASE_MODEL)
model = AutoModelForImageTextToText.from_pretrained(
    BASE_MODEL,
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
apply_vision_delta(model, MAE_DELTA)
```

Loading order matters:

```text
original Chandra -> MAE vision delta -> alignment training
```

## Freeze Policy

For the first alignment run, train only Chandra's native merger:

```python
model.requires_grad_(False)
model.model.visual.merger.requires_grad_(True)

model.config.use_cache = False
model.gradient_checkpointing_enable()
```

Do not wrap the frozen language model in `torch.no_grad()`. Although its weights
are frozen, autograd must propagate the language loss back through the language
model to the merger output.

Verify trainable parameters before launching:

```python
trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
assert trainable
assert all("visual.merger" in name for name in trainable)
print("\n".join(trainable))
```

## Chat Formatting and Labels

Use Chandra's native chat template. A training conversation should be:

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": example["prompt"]},
        ],
    },
    {
        "role": "assistant",
        "content": example["target"],
    },
]
```

Render the full conversation and the user prompt separately:

```python
user_messages = messages[:1]

prompt_text = processor.apply_chat_template(
    user_messages,
    tokenize=False,
    add_generation_prompt=True,
)
full_text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False,
)
```

Process the image with each string to obtain the exact expanded prompt length.
The image placeholder expands to a variable number of tokens, so counting text
tokens alone is incorrect:

```python
prompt_inputs = processor(
    text=[prompt_text],
    images=[image],
    return_tensors="pt",
)
inputs = processor(
    text=[full_text],
    images=[image],
    return_tensors="pt",
)

prompt_length = prompt_inputs["input_ids"].shape[1]
labels = inputs["input_ids"].clone()
labels[:, :prompt_length] = -100
labels[inputs["attention_mask"] == 0] = -100
inputs["labels"] = labels
```

Only assistant response tokens should contribute to the autoregressive loss.
Mask user text, image placeholders, padding, and the assistant-generation
prefix. In a production collator, batch examples only after computing each
sample's prompt boundary, then pad both inputs and labels consistently.

## Initial Training Configuration

Start with projector-only alignment:

```yaml
base_model: datalab-to/chandra-ocr-2
mae_delta: outputs/chandra2-mae-full/chandra_vision_delta.safetensors
train_manifest: data/alignment_train.jsonl
validation_manifest: data/alignment_validation.jsonl
output_dir: outputs/chandra2-alignment

max_pixels: 589824
batch_size: 2
gradient_accumulation_steps: 32
max_steps: 2000

merger_learning_rate: 0.0001
weight_decay: 0.01
warmup_ratio: 0.03
scheduler: cosine
max_grad_norm: 1.0
mixed_precision: bf16

vision_encoder: frozen
language_model: frozen
gradient_checkpointing: true
```

An H200 can usually use a larger microbatch, but determine it empirically with
the real resolution and target-length distribution. Keep effective batch size
between 64 and 128 initially. Larger language targets can consume more memory
than the image tower.

Recommended first schedule:

- Smoke test: 2-10 optimizer steps.
- First evaluation: 500 steps.
- Initial full alignment: 1,000-3,000 steps.
- Save every 250-500 steps.

Select checkpoints using downstream validation, not training loss alone.

Copy and edit the provided configuration:

```bash
cp docs/alignment_example.yaml configs/alignment_local.yaml
```

At minimum, set `dataset`, `mae_delta`, and `output_dir`, then run:

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch \
  --num_processes 1 \
  --num_machines 1 \
  --mixed_precision bf16 \
  --dynamo_backend no \
  train_alignment.py \
  --config configs/alignment_local.yaml
```

Run a short integration test before the full schedule:

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch \
  --num_processes 1 \
  --mixed_precision bf16 \
  train_alignment.py \
  --config configs/alignment_local.yaml \
  --max-steps 2 \
  --gradient-accumulation-steps 1 \
  --eval-every 1 \
  --save-every 2
```

Resume from a full Accelerate checkpoint with:

```bash
accelerate launch train_alignment.py \
  --config configs/alignment_local.yaml \
  --resume-from outputs/chandra2-alignment/checkpoints/step-250
```

TensorBoard logging is enabled by default. View it while training with:

```bash
tensorboard --logdir outputs/chandra2-alignment/logs --port 6006
```

Set `report_to: wandb` after running `wandb login`, use
`report_to: tensorboard,wandb` for both dashboards, or set `report_to: none` to
disable tracker logging. The JSONL metrics file is written regardless.

Targets that do not fit within `max_sequence_length` are skipped by default,
before tensors are moved to the GPU. The cumulative count is shown as
`skipped` in the progress bar and written to `metrics.jsonl`. Keep this policy
explicit in the configuration:

```yaml
max_sequence_length: 8192
overlength_policy: skip
```

Use `overlength_policy: error` when auditing a new dataset if training should
stop on the first overlength label. On one GPU, other examples in a mixed batch
are retained. During distributed training, an entire cross-rank microbatch is
dropped if any rank contains an overlength example, which keeps DDP execution
and gradient weighting synchronized.

## Optimizer

For projector-only training:

```python
from torch.optim import AdamW

optimizer = AdamW(
    model.model.visual.merger.parameters(),
    lr=1e-4,
    weight_decay=0.01,
)
```

Use BF16, cosine decay, 3% warmup, and gradient clipping at 1.0. The training
loss is the full model's assistant-only causal language-model loss:

```python
outputs = model(**inputs)
loss = outputs.loss
```

## Optional Joint Vision Alignment

If projector-only performance plateaus while validation remains below the
original Chandra baseline, run a second phase with the vision encoder unfrozen
at a much smaller learning rate:

```python
model.model.visual.requires_grad_(True)
model.model.visual.merger.requires_grad_(True)

merger_ids = {id(parameter) for parameter in model.model.visual.merger.parameters()}
vision_parameters = [
    parameter
    for parameter in model.model.visual.parameters()
    if id(parameter) not in merger_ids
]

optimizer = AdamW(
    [
        {"params": vision_parameters, "lr": 2e-6},
        {"params": model.model.visual.merger.parameters(), "lr": 5e-5},
    ],
    weight_decay=0.01,
)
```

Keep the language model frozen. Run this phase briefly, typically 500-2,000
steps, and monitor both target-domain improvement and general Chandra
regression.

## Checkpointing

Save full optimizer and scheduler state for resumption. Also export a compact
alignment delta containing the updated merger:

```python
from safetensors.torch import save_file

save_file(
    {
        f"model.visual.merger.{name}": tensor.detach().cpu().contiguous()
        for name, tensor in model.model.visual.merger.state_dict().items()
    },
    "outputs/chandra2-alignment/alignment_merger_delta.safetensors",
    metadata={"base_model": BASE_MODEL, "mae_delta": MAE_DELTA},
)
```

If the optional joint phase trains vision parameters, export those parameters
as well; a merger-only delta would not reproduce that run.

The final inference/SFT loading order is:

```text
original Chandra
  -> MAE vision delta
  -> alignment merger delta
  -> optional SFT/LoRA adapter
```

## Evaluation

Use the same held-out pages and generation settings for all models:

| Model | Purpose |
|---|---|
| Original Chandra | Baseline |
| Chandra + MAE delta | Measure representation drift |
| Chandra + MAE + alignment | Measure recovered multimodal compatibility |

Recommended metrics:

- OCR: CER, WER, normalized edit distance.
- Markdown: normalized edit distance and structural accuracy.
- Tables: TEDS and cell-text accuracy.
- Layout: block F1, IoU, and reading-order accuracy.
- Structured output: exact match, field F1, and JSON validity.

Also inspect generated pages manually. Alignment loss can decrease while the
model learns undesirable formatting shortcuts or copies noisy pseudo-labels.

## Acceptance Criteria

Proceed to document SFT only when:

1. Training and validation loss remain finite and stable.
2. The aligned model clearly improves over the unaligned MAE model.
3. OCR/Markdown quality approximately recovers the original Chandra baseline.
4. The exported deltas reload without missing or unexpected trainable keys.
5. General Chandra validation does not show unacceptable regression.

After these checks, perform document SFT with the vision encoder frozen or at a
tiny learning rate, the merger trainable, and the language model adapted with
LoRA or normal supervised fine-tuning.

## Common Failure Modes

### Loss is zero or does not change

- Confirm that some labels are not `-100`.
- Confirm that merger parameters have `requires_grad=True`.
- Confirm that merger gradients are nonzero after backward.

### Loss decreases but generation is empty

- Include the assistant end token in labels.
- Use the same chat template for training and inference.
- Check that prompt masking does not also mask target tokens.

### CUDA out of memory

- Reduce image `max_pixels`.
- Reduce microbatch size and increase accumulation.
- Reduce maximum target length.
- Enable gradient checkpointing.

### JSON targets exceed the sequence limit

- Keep `overlength_policy: skip` to exclude them without training on truncated,
  invalid JSON.
- Monitor `skipped_overlength_samples` in `metrics.jsonl`; a large count means
  the labels should be split or the data distribution reconsidered.
- Do not raise the context ceiling far beyond typical sample lengths merely to
  accommodate a few outliers, because long-context backpropagation can exhaust
  GPU memory.

### Training is slow despite a large GPU

- Profile image decoding and tokenization separately.
- Cache tokenized targets when practical.
- Avoid millions of random loose-file reads; use indexed or sharded data.
- Group examples by similar image and target lengths to reduce padding.

### The aligned model remains worse than original Chandra

- Verify that the correct MAE delta was loaded.
- Increase alignment data quality before increasing training duration.
- Try the short joint-vision phase with a very small vision learning rate.
- Compare against a direct-SFT baseline before investing in longer alignment.
