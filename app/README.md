# ParseAnything

A production-shaped FastAPI application with a responsive web frontend for
parsing images and PDFs through a MonkeyOCRv2 vLLM endpoint.

## Start

Create the local configuration first:

```bash
cp .env.example .env
```

Set `VLLM_URL` in `.env` to the OpenAI-compatible `/v1` endpoint, then run with
Docker Compose:

```bash
docker compose up --build -d
docker compose logs -f parseanything
```

Open <http://127.0.0.1:8000>. Stop the service with `docker compose down`.

For local development without Docker:

```bash
source /media/drive-2t/miniconda3/etc/profile.d/conda.sh
conda activate main
pip install -r requirements.txt
python app.py
```

API documentation is available at <http://127.0.0.1:8000/docs>.

## Configuration

| Variable | Default |
| --- | --- |
| `VLLM_URL` | `http://127.0.0.1:8888/v1` outside Docker |
| `MONKEYOCR_MODEL` | `MonkeyOCRv2` |
| `MONKEYOCR_API_KEY` | `not-required` |
| `MONKEYOCR_MAX_PIXELS` | `1003520` (official vLLM image budget) |
| `MONKEYOCR_PIPELINE_MODE` | `staged` (official layout-then-recognition flow) |
| `MONKEYOCR_KEEP_HEADER_FOOTER` | `false` |
| `MONKEYOCR_PROMPT` | Official prompt; used only in `end2end` mode |
| `MAX_UPLOAD_MB` | `30` |
| `MAX_PDF_PAGES` | `20` |
| `MAX_IMAGE_SIDE` | `3200` |
| `MIN_IMAGE_PIXELS` | `1003520` (upscale smaller OCR images) |
| `MAX_PREVIEW_SIDE` | `1200` |
| `PDF_RENDER_DPI` | `200` |
| `OCR_CONCURRENCY` | `3` |
| `OCR_TIMEOUT_SECONDS` | `300` |
| `OCR_HTTP_RETRIES` | `5` |
| `OCR_RETRY_BACKOFF_SECONDS` | `1` |
| `OCR_REPEAT_RETRIES` | `3` |
| `HOST` / `PORT` | `127.0.0.1` / `8000` (`PORT` is the published port in Compose) |

`MONKEYOCR_BASE_URL` remains accepted as a fallback for existing deployments.
The Compose example uses `host.docker.internal` so a vLLM server running on the
Docker host is reachable from the application container.

The UI prepares thumbnail previews for multi-page PDFs and TIFFs so individual
pages can be selected before OCR. The batch-size text field accepts any valid
integer up to the selected-page count; its buttons step through powers of two
and cap at that count. Like the
[official MonkeyOCRv2 Gradio demo](https://github.com/Yuliang-Liu/MonkeyOCRv2/blob/main/parsing/demo/gradio_demo.py),
PDFs are split into selected pages and rasterized as separate RGB images. Small
page and image inputs are upscaled to `MIN_IMAGE_PIXELS` before the default
pipeline detects the ordered page layout and recognizes each crop with the
prompt appropriate for text, formulas, or OTSL tables.
Picture crops are embedded directly in the Markdown. Set
`MONKEYOCR_PIPELINE_MODE` to `end2end` to use a single request per page instead.
Degenerate repeated-token responses are retried using the same progressively
higher-temperature strategy as the reference pipeline.

Bounding boxes are returned in page-pixel coordinates together with the page
dimensions. Results can be switched between sanitized Markdown, the zoomable
layout overlay, and raw Markdown. Click a detected box to edit its content or
change whether it appears in Markdown. Selected boxes can also be moved or
resized with drag handles, or adjusted using exact pixel coordinates. Saved
edits immediately update the preview, copied text, and downloadable Markdown
and structured JSON files.

## API

- `POST /api/preview` accepts `file` and returns page thumbnails.
- `POST /api/parse` accepts `file` plus optional `selected_pages` such as
  `1,3-5`, and optional `batch_size` from `1` to the number of selected pages.
  Batch size defaults to `1`; omitting `selected_pages` processes every page.
