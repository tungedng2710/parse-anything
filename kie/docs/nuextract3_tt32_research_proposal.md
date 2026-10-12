# Template-guided document extraction with NuExtract3

Research proposal · 12 October 2026 · No GPU experiments run

## Problem definition

**Input: document image(s) and a supplied JSON template. Output: that template filled with values extracted from the images.**

Learn `f(I, T) → Y`, where `I` contains page images in order, `T` specifies requested fields, structure, types and extraction rules, and `Y` is the filled JSON. Preserve keys, nesting and scalar types; instantiate repeated rows as arrays. Use null for absent scalars, empty arrays for absent lists, and the template's boolean convention for checkboxes. Each value must be supported by the image and interpreted according to the template.

Every experiment addresses this same **template-conditioned key information extraction (KIE)** task. Field instructions are derived from the supplied template. PDFs are used only to render images; their embedded text is not an input. Internal crops or image-derived OCR may support extraction while preserving the input/output contract. The central hypothesis is that schema-aligned supervision and selective visual grounding improve field accuracy, particularly for dense tables, blanks and checkboxes. This is a hypothesis, not an observed improvement.

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

## Focused extraction survey

| Approach | Research evidence | Implication for TT32 |
| --- | --- | --- |
| Template-conditioned VLM extraction | [NuExtract3](https://huggingface.co/numind/NuExtract3) is a 4B Qwen3.5-based VLM supporting template-guided JSON, field instructions and optional reasoning. | Direct baseline for the defined task. Preserve its native prompt format; publisher benchmarks do not establish Vietnamese TT32 extraction accuracy. |
| Direct image-to-structure generation | [Donut](https://arxiv.org/abs/2111.15664) generates structured document outputs directly from images without an external OCR stage. | Supports image-to-JSON supervision. NuExtract adaptation must also respect each supplied template instead of memorizing one output schema. |
| Layout-grounded extraction | [LayoutLMv3](https://arxiv.org/abs/2204.08387) combines text, image and layout representations for document understanding, including extraction tasks. | Motivates optional internal image-derived OCR/boxes for field localization and row/column association. A complete comparator still needs template-conditioned field selection and JSON assembly. |
| Efficient adaptation | [LoRA](https://arxiv.org/abs/2106.09685) trains low-rank updates; [QLoRA](https://arxiv.org/abs/2305.14314) combines adapters with a quantized base. | Start with frozen vision and language-module adapters. NF4 QLoRA is an optional memory tradeoff; TT32 accuracy and memory requirements remain unmeasured. |
| Schema-constrained extraction output | [vLLM structured outputs](https://docs.vllm.ai/en/latest/features/structured_outputs/) constrain generated JSON structure. | Can enforce keys and types derived from the input template, but cannot guarantee that extracted values match the image. |

## Proposed improvements and experiments

1. **Preserve template semantics.** Convert the custom `fields/type` KIE format into NuExtract type templates and derive field-path instructions from the supplied template's descriptions. This deterministic adapter must be independent of target labels. The descriptive `*.template.json` files are not NuExtract type templates and sometimes contain older rules; resolve those differences before freezing the benchmark. Keep dates, IDs and numeric-looking values as strings when requested; retain nulls and booleans. Compare types-only prompts against types plus descriptions.
2. **Adapt image-and-template-conditioned extraction.** Train on verified `(images, template, filled JSON)` examples using completion-only LoRA: rank 16, alpha 32, dropout 0.05, learning rate `1e-4`, three initial epochs and effective batch eight on one GPU. Tune on validation data. Include Qwen3.5 full and linear attention projections plus MLP layers; initially freeze vision. Mask prompt, image and padding tokens, supervising only the filled JSON and termination tokens. Measure whether field accuracy improves without increasing unsupported values.
3. **Teach dependence on the template.** Create training-only template variants by selecting field subsets or reordering object keys, with matching target projections and unchanged field meanings. Preserve required nesting and complete requested table-row fields; validate every derived target. Evaluate on held-out template combinations. Hypothesis: this reduces schema memorization and improves extraction of the fields actually requested.
4. **Ground difficult grids selectively.** On repaired forms 37/38 and laboratory tables, compare full-page extraction against full page plus overlapping table crops that retain time headers and row labels. Associate cells with explicit column coordinates; assemble the requested JSON by time/row keys with conflict detection. Sweep image pixel budget separately from rendering DPI: processor resizing can erase higher-DPI detail. Crop preparation and geometry-based assembly are proposed follow-up work.
5. **Control omissions and unsupported values.** Prioritize verified examples with blank fields, unchecked boxes, clipped text and intentional empty table columns. Add label-preserving scan degradation only to training data. Separately ablate template-derived JSON constraints and selective reasoning for difficult extraction cases. Score extracted values independently of structural validity.

Use a staged matrix: **B0** native image/template NuExtract3; **B1** consistent template descriptions; **B2** B1 + image/template LoRA; **B3** B2 + template variants; **B4** B3 + selective crops; **B5** B4 + constrained JSON. Study reasoning separately. Compare one change at a time with matched splits, checkpoint revision, template semantics and image budgets except for the variable under study. No scores are available yet.

## Evaluation and deliverables

The supplied pipeline excludes forms 5/37 and groups documents by form family, duplicate PDF hash and conservatively detected patient IDs/names. Seed 42 produces **32 eligible documents, 18 connected groups: 27 train / 2 validation / 3 test**. This is a debugging pilot, not a statistically reliable benchmark. Audit patient linkage using an explicit group map; collect additional independent encounters and hospitals. Separate patient-disjoint within-form extraction from unseen-form/template transfer. Keep template variants and image augmentations inside training partitions. With more groups, use grouped folds and confidence intervals resampled at the independent-group level.

Report document-level exact filled-JSON match, document/form macro and leaf micro field accuracy, missing/extra field rates, null hallucinations, checkbox accuracy and indexed table-cell accuracy. Report JSON/schema validity and median/p95 end-to-end extraction latency separately. Strict strings use Unicode NFC only; optionally add whitespace-normalized similarity as a secondary metric. Preserve row order and add time-key alignment diagnostics for dense grids. Evaluate chart-derived fields only when requested by the template and their labels/tolerances have been adjudicated. Prefer gains that persist across grouped splits without worse unsupported values or checkbox accuracy; choose settings on validation data before opening the final test set.

Implemented by default: [image-only preparation, completion-only LoRA/QLoRA, filled-JSON inference and field-level evaluation](../training/README.md), with local manifests and audit in `artifacts/tt32/`. CPU preparation and basic syntax/CLI checks were performed in conda `ai`. Training, model inference and GPU tests were **not run**. Template variants, selective crops, constrained decoding, document-level exact-match reporting and additional diagnostics remain follow-up work. The current evaluator measures generation latency; end-to-end extraction timing is also follow-up work.
