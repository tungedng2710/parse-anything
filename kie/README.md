# NuExtract Studio

A small FastAPI demo for structured extraction with a NuExtract3 model served
through vLLM's OpenAI-compatible API.

## Run

Start the NuExtract vLLM server first, then launch the demo in another terminal:

```bash
conda activate main
export NUEXTRACT_BASE_URL=http://127.0.0.1:8895/v1
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`. Upload one PDF/PNG/JPEG/WebP document and one JSON
template. The result can be inspected as nested key-value data or raw JSON and
downloaded from the browser.

Useful environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `NUEXTRACT_BASE_URL` | `http://127.0.0.1:8895/v1` | vLLM API URL |
| `NUEXTRACT_MODEL` | `numind/NuExtract3` | Served model name |
| `NUEXTRACT_API_KEY` | `EMPTY` | vLLM API key |
| `NUEXTRACT_MAX_TOKENS` | `8192` | Maximum generated tokens |
| `NUEXTRACT_PDF_DPI` | `170` | PDF rendering resolution |
| `NUEXTRACT_MAX_DOCUMENT_MB` | `40` | Upload-size limit |
| `NUEXTRACT_MAX_PDF_PAGES` | `30` | Page-count limit |

## Test

```bash
conda activate main
python -m pytest -q
```

## Medical 9 forms benchmark

With both vLLM and the FastAPI demo running, generate all predictions:

```bash
conda activate main
python data/medial_9_forms/predict.py --force --concurrency 2
```

Evaluate the saved JSON files and optionally write a JSON report:

```bash
python data/medial_9_forms/eval.py \
  --report data/medial_9_forms/evaluation_report.json \
  --show-errors 3
```

The evaluator uses normalized indel similarity for string leaves and exact match
for non-string leaves. Missing and extra leaf paths score zero. It reports both
macro (equal weight per document) and micro (equal weight per leaf) metrics.
