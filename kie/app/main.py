"""FastAPI backend for the NuExtract3 document extraction demo."""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

import fitz
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI
from PIL import Image, UnidentifiedImageError


LOGGER = logging.getLogger("nuextract-demo")
APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"

MODEL_ID = os.getenv("NUEXTRACT_MODEL", "numind/NuExtract3")
BASE_URL = os.getenv("NUEXTRACT_BASE_URL", "http://127.0.0.1:8895/v1")
API_KEY = os.getenv("NUEXTRACT_API_KEY", "EMPTY")
REQUEST_TIMEOUT = float(os.getenv("NUEXTRACT_TIMEOUT", "600"))
MAX_OUTPUT_TOKENS = int(os.getenv("NUEXTRACT_MAX_TOKENS", "16000"))
PDF_DPI = int(os.getenv("NUEXTRACT_PDF_DPI", "170"))
MAX_DOCUMENT_BYTES = int(os.getenv("NUEXTRACT_MAX_DOCUMENT_MB", "40")) * 1024 * 1024
MAX_TEMPLATE_BYTES = int(os.getenv("NUEXTRACT_MAX_TEMPLATE_KB", "512")) * 1024
MAX_PDF_PAGES = int(os.getenv("NUEXTRACT_MAX_PDF_PAGES", "30"))

SUPPORTED_IMAGE_TYPES = {
    "image/jpeg": "jpeg",
    "image/png": "png",
    "image/webp": "webp",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.nuextract_client = AsyncOpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
        timeout=REQUEST_TIMEOUT,
    )
    yield
    await app.state.nuextract_client.close()


app = FastAPI(
    title="NuExtract Studio",
    description="Upload a document and JSON template for structured extraction.",
    version="1.0.0",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health(request: Request) -> dict[str, Any]:
    model_server = "unavailable"
    try:
        await request.app.state.nuextract_client.models.list()
        model_server = "ready"
    except Exception:  # The health endpoint must remain available for the UI.
        LOGGER.debug("Model health check failed", exc_info=True)

    return {
        "status": "ok",
        "model_server": model_server,
        "model": MODEL_ID,
        "base_url": BASE_URL,
    }


async def read_limited(upload: UploadFile, limit: int, label: str) -> bytes:
    data = await upload.read(limit + 1)
    if not data:
        raise HTTPException(status_code=400, detail=f"The {label} file is empty.")
    if len(data) > limit:
        raise HTTPException(
            status_code=413,
            detail=f"The {label} file exceeds the configured size limit.",
        )
    return data


def parse_template(data: bytes) -> dict[str, Any]:
    try:
        decoded = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=400, detail="The template must be UTF-8 encoded JSON."
        ) from exc

    try:
        template = json.loads(decoded)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid template JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}.",
        ) from exc

    if not isinstance(template, dict) or not template:
        raise HTTPException(
            status_code=400,
            detail="The template must be a non-empty JSON object.",
        )
    return template


def encode_png(png_bytes: bytes) -> str:
    encoded = base64.b64encode(png_bytes).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def prepare_image(data: bytes) -> list[str]:
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            if image.width < 1 or image.height < 1:
                raise ValueError("invalid image dimensions")
            rgb_image = image.convert("RGB")
            output = io.BytesIO()
            rgb_image.save(output, format="PNG", optimize=True)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=400, detail="The uploaded image could not be decoded."
        ) from exc
    return [encode_png(output.getvalue())]


def prepare_pdf(data: bytes) -> list[str]:
    try:
        with fitz.open(stream=data, filetype="pdf") as document:
            if document.needs_pass:
                raise HTTPException(
                    status_code=400, detail="Password-protected PDFs are not supported."
                )
            if document.page_count < 1:
                raise HTTPException(status_code=400, detail="The PDF has no pages.")
            if document.page_count > MAX_PDF_PAGES:
                raise HTTPException(
                    status_code=400,
                    detail=f"PDFs are limited to {MAX_PDF_PAGES} pages.",
                )

            page_images: list[str] = []
            for page in document:
                pixmap = page.get_pixmap(dpi=PDF_DPI, alpha=False)
                page_images.append(encode_png(pixmap.tobytes("png")))
            return page_images
    except HTTPException:
        raise
    except (fitz.FileDataError, RuntimeError, ValueError) as exc:
        raise HTTPException(
            status_code=400, detail="The uploaded PDF could not be decoded."
        ) from exc


def prepare_document(
    filename: str | None, content_type: str | None, data: bytes
) -> tuple[list[str], str]:
    suffix = Path(filename or "").suffix.lower()
    normalized_type = (content_type or "").split(";", maxsplit=1)[0].lower()

    if normalized_type == "application/pdf" or suffix == ".pdf":
        return prepare_pdf(data), "pdf"
    if normalized_type in SUPPORTED_IMAGE_TYPES or suffix in {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
    }:
        return prepare_image(data), "image"
    raise HTTPException(
        status_code=415,
        detail="Unsupported document type. Upload a PDF, PNG, JPEG, or WebP file.",
    )


def parse_model_json(raw_result: str) -> Any:
    result = raw_result.strip()
    if "</think>" in result:
        result = result.split("</think>", maxsplit=1)[1].strip()

    if result.startswith("```"):
        first_newline = result.find("\n")
        if first_newline >= 0:
            result = result[first_newline + 1 :]
        if result.rstrip().endswith("```"):
            result = result.rstrip()[:-3].rstrip()

    try:
        return json.loads(result)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        starts = [position for position in (result.find("{"), result.find("[")) if position >= 0]
        if starts:
            try:
                parsed, _ = decoder.raw_decode(result[min(starts) :])
                return parsed
            except json.JSONDecodeError:
                pass
        raise HTTPException(
            status_code=502,
            detail="NuExtract returned an incomplete or invalid JSON response. Try again or increase the output-token limit.",
        )


@app.post("/api/extract")
async def extract(
    request: Request,
    document: Annotated[UploadFile, File(description="PDF or image document")],
    template: Annotated[UploadFile, File(description="NuExtract JSON template template")],
    instructions: Annotated[str, Form()] = "",
    thinking: Annotated[bool, Form()] = False,
) -> dict[str, Any]:
    document_data = await read_limited(document, MAX_DOCUMENT_BYTES, "document")
    template_data = await read_limited(template, MAX_TEMPLATE_BYTES, "template")
    extraction_template = parse_template(template_data)
    image_urls, document_kind = prepare_document(
        document.filename, document.content_type, document_data
    )

    content = [
        {"type": "image_url", "image_url": {"url": image_url}}
        for image_url in image_urls
    ]
    chat_template_kwargs: dict[str, Any] = {
        "template": json.dumps(extraction_template, ensure_ascii=False),
        "enable_thinking": thinking,
    }
    if instructions.strip():
        chat_template_kwargs["instructions"] = instructions.strip()

    started = time.perf_counter()
    try:
        response = await request.app.state.nuextract_client.chat.completions.create(
            model=MODEL_ID,
            temperature=0.6 if thinking else 0.2,
            max_tokens=MAX_OUTPUT_TOKENS,
            messages=[{"role": "user", "content": content}],
            extra_body={"chat_template_kwargs": chat_template_kwargs},
        )
    except APITimeoutError as exc:
        raise HTTPException(
            status_code=504, detail="NuExtract timed out while processing the document."
        ) from exc
    except APIConnectionError as exc:
        raise HTTPException(
            status_code=503,
            detail="Could not connect to the NuExtract model server.",
        ) from exc
    except APIStatusError as exc:
        LOGGER.warning("NuExtract returned HTTP %s", exc.status_code)
        raise HTTPException(
            status_code=502,
            detail=f"NuExtract rejected the request (HTTP {exc.status_code}).",
        ) from exc

    raw_result = response.choices[0].message.content or ""
    result = parse_model_json(raw_result)
    elapsed_ms = round((time.perf_counter() - started) * 1000)
    usage = response.usage

    return {
        "result": result,
        "metadata": {
            "filename": document.filename or "document",
            "document_type": document_kind,
            "pages": len(image_urls),
            "elapsed_ms": elapsed_ms,
            "model": MODEL_ID,
            "completion_tokens": getattr(usage, "completion_tokens", None),
        },
    }
