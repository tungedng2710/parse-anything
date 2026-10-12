# TT32 NuExtract3 adaptation pipeline

Research task: document image(s) plus a supplied JSON template produce filled
JSON. Use the default `image` input mode for experiments in the proposal;
PDFs are rendered to images during preparation.

Run commands from the `kie/` repository directory. This pipeline is separate from
the FastAPI demo and uses local Hugging Face weights rather than the vLLM endpoint.
The code was researched against NuExtract3 and Transformers 5.19 APIs. **No training,
model inference, GPU checks or test suites have been run on this device.** Only CPU
preparation and syntax inspection have been performed in conda `ai`.

## Prepare data on CPU

```bash
conda activate ai
python -m training.prepare --data-root data/v2_TT32_forms --output artifacts/tt32
```

Preparation resolves Unicode-normalized paths through `kie_schema_map.json`,
checks every mapped label against its schema, converts custom KIE schemas to
NuExtract type templates plus current field instructions, and renders ordered
page PNGs at 170 DPI. It writes `train.jsonl`, `validation.jsonl`, `test.jsonl`
and `audit.json`. There is no training or model download in this step.

The initial snapshot has 34 mapped documents and five unmapped PDFs. Forms 5 and
37 are excluded by default because the local review identifies unresolved chart
policy and/or incorrect labels. The remaining 32 documents produce 18 connected
groups, with 27/2/3 documents in train/validation/test for seed 42.

Splits keep identical PDF hashes and form families together, and link patient IDs
and names detected in patient blocks. This linkage is heuristic and can miss
aliases or over-group shared names. Use `--group-map path/to/groups.json` to add
audited patient/encounter grouping: keys are NFC-normalized PDF paths relative to
the data root, values are shared group IDs. Additional grouping joins the existing
family/patient components; it does not weaken them. Inspect `audit.json` before
reporting results. The split measures unseen-form transfer; the small holdout is
suitable for a pilot only.

Manifests contain absolute image paths. Prepare again on the GPU host after moving
the source data. Alternatively, preserve the same absolute image location. An
existing output requires `--overwrite`; use a fresh output for a new experiment
to avoid mixing datasets. Include unresolved forms only after adjudication:

```bash
python -m training.prepare --output artifacts/tt32-reviewed --exclude-forms
```

## Train on a CUDA GPU host

Install an appropriate CUDA PyTorch build for the host, followed by:

```bash
python -m pip install -r training/requirements.txt
python -m training.train \
  --data artifacts/tt32 \
  --output artifacts/nuextract3-tt32-lora \
  --revision c99dc8f5641b866aa0192b6ea78f84bf9f3535f1
```

The revision above is the NuExtract3 commit inspected on 2026-10-12. The CLI
defaults to `main`; pass an immutable revision to reproduce a baseline comparison.

Default settings: rank 16, alpha 32, dropout 0.05, learning rate `1e-4`, three
epochs, batch size one and gradient accumulation eight. Vision and the output
head remain frozen. LoRA targets language-model full-attention projections,
linear-attention projections and MLP projections. The native NuExtract chat
template includes its empty non-thinking prefix. Loss covers only the JSON
completion and termination tokens; prompt, image and padding tokens are masked.
Prompt/completion token alignment is checked after vision-token expansion.

Images are limited to 1,605,632 pixels per page by the processor. `--max-length`
rejects oversized sequences instead of truncating image or label tokens.
Gradient checkpointing is enabled and inference caches are disabled during SFT.
Validation loss selects the saved adapter; test data is only read to check split
overlap and is never used for training or checkpoint selection. Best validation
loss is a starting point; select task-metric settings using validation predictions.
Actual GPU memory and compatibility have not been measured.

Optional NF4 QLoRA requires a supported CUDA/bitsandbytes installation:

```bash
python -m pip install 'bitsandbytes>=0.49'
python -m training.train --data artifacts/tt32 \
  --output artifacts/nuextract3-tt32-qlora --qlora \
  --revision c99dc8f5641b866aa0192b6ea78f84bf9f3535f1
```

Use `--max-pixels` to reduce memory or study resolution. Do not interpret an OOM
as a reason to silently truncate examples. Resume with `--resume` pointing to a
Trainer `checkpoint-*` directory, keeping the original data and arguments.

Each run saves Trainer checkpoints, `adapter/` (PEFT adapter and processor), and
`run_config.json` with arguments, library versions, resolved base revision,
target modules and selected checkpoint. This is an adapter artifact, not a merged
full-model export; speculative MTP serving is outside this pipeline.

## Predict and evaluate

Generate baseline and adapted predictions using matching image budgets and
the same immutable base revision. Both commands use greedy decoding; this is a
controlled baseline, distinct from the publisher's sampled serving defaults.
The default baseline includes descriptions (B1 in the proposal). Add
`--no-field-instructions` and a separate output directory for the types-only B0
ablation; compare B1 against the adapted model to isolate the training effect.

```bash
python -m training.predict --manifest artifacts/tt32/test.jsonl \
  --output artifacts/predictions-baseline \
  --revision c99dc8f5641b866aa0192b6ea78f84bf9f3535f1
python -m training.predict --manifest artifacts/tt32/test.jsonl \
  --adapter artifacts/nuextract3-tt32-lora/adapter \
  --output artifacts/predictions-lora
python -m training.evaluate --manifest artifacts/tt32/test.jsonl \
  --predictions artifacts/predictions-baseline --report artifacts/baseline-report.json
python -m training.evaluate --manifest artifacts/tt32/test.jsonl \
  --predictions artifacts/predictions-lora --report artifacts/lora-report.json
```

During model selection use `validation.jsonl`; reserve the final test comparison
until settings are fixed. Adapted inference reads the saved input mode, pixel
budget and base revision automatically. QLoRA adapters may be evaluated against
a full-precision base or a quantized base (`--qlora`); keep that choice identical
when comparing runs and report it. Each prediction retains raw `.txt`, parsed
`.json` where valid, and generation timing/token/schema metadata. Invalid JSON
remains a failure, without silent JSON repairs. Existing prediction directories
require `--overwrite`.

The CPU evaluator scores all manifest documents, including missing/invalid
predictions, with missing and extra paths penalized. It reports document and form
macro exact accuracy, leaf micro exact accuracy, JSON/schema validity, separate
null/checkbox/array/scalar accuracy, blank hallucinations and generation latency.
Strings normalize Unicode NFC only; arrays retain indices. Null and checkbox
categories take precedence over array cells. Missing predictions reduce accuracy;
blank hallucination rate separately counts explicitly filled reference-null fields.

Research scope and proposed follow-up experiments are in
[the short proposal](../docs/nuextract3_tt32_research_proposal.md).
