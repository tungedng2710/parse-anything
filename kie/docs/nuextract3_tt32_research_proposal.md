# Incremental improvements to NuExtract3 for TT32 document extraction

Research proposal · 12 October 2026 · No GPU experiments run

## Problem definition

**Input: document image(s) and a supplied JSON template. Output: that template filled with values extracted from the images.**

Use NuExtract3 to compute `f(I, T) → Y`, where `I` contains page images in order, `T` specifies requested fields, structure, types and extraction rules, and `Y` is the filled JSON. Preserve keys, nesting and scalar types; instantiate repeated rows as arrays. Use null for absent scalars, empty arrays for absent lists, and the template's boolean convention for checkboxes. Each value must be supported by the image and interpreted according to the template.

**NuExtract3 is the fixed baseline and the model being improved.** The work consists of measuring its errors on TT32 and applying targeted changes to its prompt, image preprocessing, fine-tuning data and decoding. Keep the architecture and image/template-to-JSON interface throughout. Field instructions come from the supplied template; PDFs are used only to render images. The central hypothesis is that these incremental changes improve Vietnamese field accuracy, table alignment and missing-value handling. No gains have been measured yet.

## Dataset evidence

Local inspection of `data/v2_TT32_forms` found:

| Item | Observed count |
| --- | ---: |
| PDFs / mapped PDF–schema–label pairs | 39 / 34 |
| Distinct mapped form folders | 33 |
| Pages in mapped documents | 43: 25 one-page and 9 two-page documents |
| Label leaves | 1,825 strings; 590 nulls; 216 booleans |
| Empty arrays | 6 |

The five unmapped PDFs are form 37 sheets 2–3 and form 39 sheets T2–T4. Labels use two naming conventions (`*.data.json` and `*.json`); filenames also differ between Unicode NFC and NFD. The schema map is the authoritative pairing source. The local report says “31 forms”; the current mapped files occupy **33 folders**, so its summary should not substitute for an inventory.

These are GLM prelabels with reported human review, rather than independently established gold labels. The [local review](../data/v2_TT32_forms/REPORT_KIE_PRELABEL.md) identifies 20 incorrect cells in form 37 and unsettled chart-value policy in forms 5 and 37. I visually inspected form 37's dense time-column grid and form 48's mostly blank consent fields. Three documents have clipped printed text: completing that text would violate their extraction rules. Exclude forms 5 and 37 from initial SFT; correct and adjudicate them before using them as a challenge benchmark.

## Survey of improvements applicable to NuExtract3

| Approach | Research evidence | Implication for TT32 |
| --- | --- | --- |
| Native extraction controls | The [NuExtract3 model card](https://huggingface.co/numind/NuExtract3) documents template types, field instructions, non-thinking/thinking modes and image input. | First improve how the existing model receives the supplied template and document. Preserve its native chat format. |
| Parameter-efficient fine-tuning | [LoRA](https://arxiv.org/abs/2106.09685) trains low-rank parameter updates while retaining the base model. | Apply supervised adaptation directly to NuExtract3 using verified TT32 image/template/JSON examples. |
| Memory-efficient fine-tuning | [QLoRA](https://arxiv.org/abs/2305.14314) combines adapters with a quantized base. | Optional implementation choice when GPU memory limits LoRA training; measure any accuracy tradeoff on the same task. |
| Constrained JSON decoding | [vLLM structured outputs](https://docs.vllm.ai/en/latest/features/structured_outputs/) constrain generated JSON structure. | Add template-derived structural constraints to NuExtract3 inference. Measure value accuracy separately because valid JSON can contain incorrect values. |

## Proposed improvements and experiments

Start by running the unmodified checkpoint and categorizing actual errors: wrong characters, incorrect nulls/checkboxes, wrong table columns, and malformed JSON. Prioritize changes against those measured errors. The local review describes label quality; it does not establish NuExtract3's error distribution.

1. **Refine the extraction prompt.** Preserve the supplied template's meaning while converting the custom `fields/type` representation into NuExtract-compatible types and field-path instructions. Resolve differences between older `*.template.json` descriptions and current schemas before freezing evaluation. Test concise rules for verbatim transcription, leading zeros, unchecked boxes, blank cells and row order. Keep all instructions derivable from the input template; do not use target labels to construct prompts. Expected benefit: fewer formatting and missing-value errors.
2. **Tune full-page image detail.** Sweep rendering DPI and the processor's pixel budget separately, using the same NuExtract3 checkpoint and prompts. Higher-DPI rendering helps only if processor resizing preserves that detail. Choose the smallest budget that improves small text and dense table cells on validation data. Expected benefit: better readability without adding extraction components.
3. **Fine-tune NuExtract3 on verified TT32 examples.** Use `(images, template, filled JSON)` triples and completion-only LoRA. Start with rank 16, alpha 32, dropout 0.05, learning rate `1e-4`, three epochs and effective batch eight; select settings on validation data. Adapt its full/linear attention and MLP projections, initially freezing vision. Mask prompt, image and padding tokens. Within the training partition, prioritize corrected examples of observed errors, including blanks, checkboxes and dense tables; consider mild label-preserving scan degradation after the basic adapter is evaluated. Expected benefit: improved Vietnamese transcription and field association using the existing model.
4. **Tighten existing inference controls.** Evaluate template-derived JSON constraints when structural errors occur. Test NuExtract3's native thinking mode separately on validation cases where field association remains difficult, accounting for added latency. Preserve failed predictions in evaluation instead of silently repairing their values. Expected benefit: fewer structural failures, with any value-accuracy gain measured independently.

Freeze **B0** as the unmodified NuExtract3 checkpoint with the current native image/template prompt, rendering settings and decoding. Apply the same necessary schema conversion to every run. Record that configuration before improvements; a types-only prompt is an optional diagnostic, not the primary baseline.

| Experiment | Change to NuExtract3 | Comparison |
| --- | --- | --- |
| P | Refined template-derived instructions | B0 + P versus B0 |
| R | Full-page image-resolution budget | B0 + R versus B0 |
| F | TT32 LoRA adapter | B0 + F versus B0, with identical prompts/images |
| D | JSON-constrained decoding | B0 + D versus B0 |
| Combined | Improvements supported by validation results | Combined versus B0; remove each component to check its contribution |

Keep splits, checkpoint revision, template semantics and evaluation fixed. For F, match base-weight precision between baseline and adapted inference. Study native thinking mode separately if needed. This is an empirical improvement plan for NuExtract3.

## Evaluation and deliverables

The supplied pipeline excludes forms 5/37 and groups documents by form family, duplicate PDF hash and conservatively detected patient IDs/names. Seed 42 produces **32 eligible documents, 18 connected groups: 27 train / 2 validation / 3 test**. This is a debugging pilot, not a statistically reliable benchmark. Audit patient linkage using an explicit group map and collect additional independent encounters. Compare every improvement with B0 on the same held-out documents. Keep image augmentations and error-focused sampling inside training partitions. With more groups, use repeated grouped evaluation and confidence intervals resampled at the independent-group level.

Report document-level exact filled-JSON match, document/form macro and leaf micro field accuracy, missing/extra field rates, null hallucinations, checkbox accuracy and indexed table-cell accuracy. Report JSON/schema validity and median/p95 end-to-end extraction latency separately. Strict strings use Unicode NFC only; optionally add whitespace-normalized similarity as a secondary metric. Preserve row order and add time-key alignment diagnostics for dense grids. Evaluate chart-derived fields only when requested by the template and their labels/tolerances have been adjudicated. Prefer gains that persist across grouped splits without worse unsupported values or checkbox accuracy; choose settings on validation data before opening the final test set.

Implemented: [NuExtract3 image-only preparation, completion-only LoRA/QLoRA, baseline/adapted inference and field-level evaluation](../training/README.md), with local manifests and audit in `artifacts/tt32/`. Existing prompt and pixel-budget controls support initial ablations. Constrained decoding, reasoning ablations, document-level exact-match reporting and end-to-end timing remain follow-up implementation. CPU preparation and basic syntax/CLI checks were performed in conda `ai`; training, model inference and GPU tests were **not run**.
