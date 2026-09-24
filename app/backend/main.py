from __future__ import annotations

import ast
import asyncio
import base64
import math
import os
import re
from contextlib import asynccontextmanager
from html import escape
from io import BytesIO
from pathlib import Path
from time import perf_counter

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = ROOT / "frontend"
SAMPLE_IMAGE = ROOT / "test_samples" / "page-62.png"

# Local development reads app/.env. In containers, Compose injects the same
# variables into the process; existing environment variables take precedence.
load_dotenv(ROOT / ".env", override=False)

API_BASE_URL = os.getenv(
    "VLLM_URL",
    os.getenv("MONKEYOCR_BASE_URL", "http://127.0.0.1:8888/v1"),
).rstrip("/")
MODEL = os.getenv("MONKEYOCR_MODEL", "MonkeyOCRv2")
END2END_PROMPT = os.getenv(
    "MONKEYOCR_PROMPT",
    "List the document elements in reading order, including their categories, "
    "coordinates, and the content of each element.",
)
PIPELINE_MODE = os.getenv("MONKEYOCR_PIPELINE_MODE", "staged").strip().lower()
if PIPELINE_MODE not in {"staged", "end2end"}:
    PIPELINE_MODE = "staged"
KEEP_HEADER_FOOTER = os.getenv(
    "MONKEYOCR_KEEP_HEADER_FOOTER", "false"
).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "30")) * 1024 * 1024
MAX_PAGES = int(os.getenv("MAX_PDF_PAGES", "20"))
MAX_IMAGE_SIDE = int(os.getenv("MAX_IMAGE_SIDE", "3200"))
MIN_IMAGE_PIXELS = max(0, int(os.getenv("MIN_IMAGE_PIXELS", "1003520")))
MAX_PREVIEW_SIDE = int(os.getenv("MAX_PREVIEW_SIDE", "1200"))
PDF_RENDER_DPI = max(72, int(os.getenv("PDF_RENDER_DPI", "200")))
MODEL_MAX_PIXELS = max(1, int(os.getenv("MONKEYOCR_MAX_PIXELS", "1003520")))
OCR_CONCURRENCY = int(os.getenv("OCR_CONCURRENCY", "3"))
REQUEST_TIMEOUT = float(os.getenv("OCR_TIMEOUT_SECONDS", "300"))
HTTP_MAX_RETRIES = max(0, int(os.getenv("OCR_HTTP_RETRIES", "5")))
HTTP_RETRY_BACKOFF = max(0.0, float(os.getenv("OCR_RETRY_BACKOFF_SECONDS", "1")))
REPEAT_MAX_RETRIES = max(0, int(os.getenv("OCR_REPEAT_RETRIES", "3")))

MONKEYOCR_PROMPTS = {
    "Caption": "Please output the text content from the image.",
    "List-item": "Please output the text content from the image.",
    "Page-footer": "Please output the text content from the image.",
    "Page-header": "Please output the text content from the image.",
    "Section-header": "Please output the text content from the image.",
    "Text": "Please output the text content from the image.",
    "Title": "Please output the text content from the image.",
    "Formula": "Please write out the expression of the formula in the image using LaTeX format.",
    "Table": "Please extract the table from the image and represent it in OTSL format.",
    "LAYOUT": "Please output the categories and coordinates of the document elements in reading order.",
}

IMAGE_TYPES = {"image/png", "image/jpeg", "image/webp", "image/bmp", "image/tiff"}
PDF_TYPE = "application/pdf"
ALLOWED_TYPES = IMAGE_TYPES | {PDF_TYPE}


class LayoutElement(BaseModel):
    bbox: list[float] = Field(min_length=4, max_length=4)
    label: str
    content: str
    included_in_markdown: bool = True


class PageResult(BaseModel):
    page_number: int
    markdown: str
    raw: str
    elements: list[LayoutElement]
    image_url: str
    image_width: int
    image_height: int


class ParseResponse(BaseModel):
    filename: str
    page_count: int
    source_page_count: int
    selected_pages: list[int]
    batch_size: int
    elapsed_seconds: float
    content: str
    pages: list[str]
    page_results: list[PageResult]


class PagePreview(BaseModel):
    page_number: int
    image_url: str


class PreviewResponse(BaseModel):
    filename: str
    page_count: int
    pages: list[PagePreview]


class HealthResponse(BaseModel):
    status: str
    model: str
    upstream: str
    pipeline: str


def _normalize_image(
    image: Image.Image,
    max_side: int = MAX_IMAGE_SIDE,
    min_pixels: int = MIN_IMAGE_PIXELS,
) -> Image.Image:
    """Orient and resize a raster while preserving its aspect ratio."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    max_side = max(1, max_side)
    scale = 1.0
    if max(width, height) > max_side:
        scale = max_side / max(width, height)
    elif min_pixels > 0 and width * height < min_pixels:
        scale = min(
            math.sqrt(min_pixels / (width * height)),
            max_side / max(width, height),
        )
    if not math.isclose(scale, 1.0):
        target = (
            max(1, min(max_side, math.ceil(width * scale))),
            max(1, min(max_side, math.ceil(height * scale))),
        )
        image = image.resize(target, Image.Resampling.LANCZOS)
    return image


def _prepare_model_image(
    image: Image.Image,
    *,
    min_pixels: int = 0,
    max_pixels: int = MODEL_MAX_PIXELS,
) -> Image.Image:
    """Match the pixel budget used by the official MonkeyOCR server client."""
    width, height = image.size
    pixels = width * height
    scale = 1.0
    rounding = round
    if min_pixels > 0 and pixels < min_pixels:
        scale = math.sqrt(min_pixels / pixels)
        rounding = math.ceil
    if max_pixels > 0 and pixels * scale * scale > max_pixels:
        scale = math.sqrt(max_pixels / pixels)
        rounding = math.floor
    if math.isclose(scale, 1.0):
        return image
    target = (
        max(1, int(rounding(width * scale))),
        max(1, int(rounding(height * scale))),
    )
    return image.resize(target, Image.Resampling.LANCZOS)


def _image_page_count(data: bytes) -> int:
    try:
        with Image.open(BytesIO(data)) as source:
            # JPEG/MPO and animated WebP can expose auxiliary frames that are not
            # document pages. Only TIFF frames use the multi-page image workflow.
            page_count = (
                int(getattr(source, "n_frames", 1)) if source.format == "TIFF" else 1
            )
            if page_count > MAX_PAGES:
                raise HTTPException(
                    status_code=422,
                    detail=f"The image has {page_count} frames; the limit is {MAX_PAGES}.",
                )
            return page_count
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=422, detail="The uploaded image is invalid or corrupted."
        ) from exc


def _render_image_pages(
    data: bytes,
    page_numbers: list[int],
    max_side: int,
    min_pixels: int = MIN_IMAGE_PIXELS,
) -> list[tuple[int, Image.Image]]:
    try:
        with Image.open(BytesIO(data)) as source:
            rendered = []
            for page_number in page_numbers:
                source.seek(page_number - 1)
                rendered.append(
                    (
                        page_number,
                        _normalize_image(source.copy(), max_side, min_pixels),
                    )
                )
            return rendered
    except (UnidentifiedImageError, OSError, EOFError, ValueError) as exc:
        raise HTTPException(
            status_code=422, detail="The uploaded image is invalid or corrupted."
        ) from exc


def _open_pdf(data: bytes):
    try:
        import pypdfium2 as pdfium

        document = pdfium.PdfDocument(data)
        page_count = len(document)
        if page_count == 0:
            document.close()
            raise HTTPException(status_code=422, detail="The PDF contains no pages.")
        if page_count > MAX_PAGES:
            document.close()
            raise HTTPException(
                status_code=422,
                detail=f"The PDF has {page_count} pages; the limit is {MAX_PAGES}.",
            )
        return document
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail="The uploaded PDF is invalid, encrypted, or unsupported.",
        ) from exc


def _render_pdf_pages(
    data: bytes,
    page_numbers: list[int],
    max_side: int,
    min_pixels: int = MIN_IMAGE_PIXELS,
) -> list[tuple[int, Image.Image]]:
    """Split selected PDF pages and rasterize each one as an RGB image."""
    document = _open_pdf(data)
    try:
        rendered: list[tuple[int, Image.Image]] = []
        for page_number in page_numbers:
            page = document[page_number - 1]
            try:
                width, height = page.get_size()
                scale = min(PDF_RENDER_DPI / 72, max_side / max(width, height))
                bitmap = page.render(scale=scale)
                try:
                    image = bitmap.to_pil().convert("RGB")
                finally:
                    bitmap.close()
                rendered.append(
                    (
                        page_number,
                        _normalize_image(image, max_side, min_pixels),
                    )
                )
            finally:
                page.close()
        return rendered
    finally:
        document.close()


def _pdf_page_count(data: bytes) -> int:
    document = _open_pdf(data)
    try:
        return len(document)
    finally:
        document.close()


def _image_data_uri(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=False, compress_level=3)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _preview_data_uri(image: Image.Image) -> str:
    preview = image.copy()
    if max(preview.size) > MAX_PREVIEW_SIDE:
        preview.thumbnail(
            (MAX_PREVIEW_SIDE, MAX_PREVIEW_SIDE), Image.Resampling.LANCZOS
        )
    buffer = BytesIO()
    preview.save(buffer, format="JPEG", quality=82, optimize=True)
    preview.close()
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _parse_page_selection(selection: str | None, page_count: int) -> list[int]:
    if not selection or not selection.strip():
        return list(range(1, page_count + 1))

    selected: set[int] = set()
    try:
        for token in selection.split(","):
            token = token.strip()
            if not token:
                continue
            if "-" in token:
                start_text, end_text = token.split("-", 1)
                start, end = int(start_text), int(end_text)
                if start > end:
                    raise ValueError
                selected.update(range(start, end + 1))
            else:
                selected.add(int(token))
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail="Invalid page selection. Use page numbers such as 1,3-5.",
        ) from exc

    if not selected:
        raise HTTPException(status_code=422, detail="Select at least one page to OCR.")
    invalid = sorted(page for page in selected if page < 1 or page > page_count)
    if invalid:
        raise HTTPException(
            status_code=422,
            detail=f"Selected page {invalid[0]} is outside this {page_count}-page document.",
        )
    return sorted(selected)


def _validate_batch_size(batch_size: int, selected_page_count: int) -> int:
    if batch_size < 1 or batch_size > selected_page_count:
        raise HTTPException(
            status_code=422,
            detail=f"Batch size must be between 1 and {selected_page_count}.",
        )
    return batch_size


def _strip_code_fence(content: str) -> str:
    stripped = content.strip()
    match = re.fullmatch(
        r"```(?:json|python)?\s*(.*?)\s*```", stripped, re.DOTALL | re.IGNORECASE
    )
    return match.group(1).strip() if match else stripped


def _extract_balanced_blocks(text: str, left: str, right: str) -> list[str]:
    blocks: list[str] = []
    depth = 0
    start = -1
    for index, char in enumerate(text):
        if char == left:
            if depth == 0:
                start = index
            depth += 1
        elif char == right and depth > 0:
            depth -= 1
            if depth == 0 and start != -1:
                blocks.append(text[start : index + 1])
                start = -1
    return blocks


def _extract_tolerant_list_blocks(text: str) -> list[str]:
    blocks = _extract_balanced_blocks(text, "[", "]")
    first = text.find("[")
    if first != -1:
        tail = text[first:].strip()
        missing = tail.count("[") - tail.count("]")
        if tail and missing > 0:
            blocks.append(tail + ("]" * missing))
    return list(dict.fromkeys(blocks))


def _extract_tolerant_dict_blocks(text: str) -> list[str]:
    blocks = _extract_balanced_blocks(text, "{", "}")
    for index, char in enumerate(text):
        if char != "{":
            continue
        depth = 0
        end = None
        for cursor in range(index, len(text)):
            if text[cursor] == "{":
                depth += 1
            elif text[cursor] == "}":
                depth -= 1
                if depth == 0:
                    end = cursor + 1
                    break
        blocks.append(
            text[index:end] if end is not None else text[index:] + ("}" * max(depth, 1))
        )
    return list(dict.fromkeys(blocks))


def _normalize_layout_item(item: object, include_content: bool) -> dict | None:
    if not isinstance(item, dict):
        return None
    bbox = item.get("bbox") or item.get("box") or item.get("coordinates")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        coords = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return None
    label = str(item.get("label") or item.get("category") or item.get("type") or "Text")
    normalized = {"bbox": coords, "label": label.strip() or "Text"}
    if include_content:
        normalized["content"] = str(
            item.get("content") or item.get("text") or ""
        ).strip()
    return normalized


def _parse_tolerant_items(content: str, include_content: bool) -> list[dict]:
    text = _strip_code_fence(content)
    if not text:
        return []

    def normalize_list(value: object) -> list[dict]:
        if not isinstance(value, list):
            return []
        return [
            item
            for raw in value
            if (item := _normalize_layout_item(raw, include_content))
        ]

    try:
        parsed = normalize_list(ast.literal_eval(text))
        if parsed:
            return parsed
    except (SyntaxError, ValueError, TypeError, MemoryError, RecursionError):
        pass

    best: list[dict] = []
    for block in _extract_tolerant_list_blocks(text):
        try:
            current = normalize_list(ast.literal_eval(block))
            if len(current) > len(best):
                best = current
        except (SyntaxError, ValueError, TypeError, MemoryError, RecursionError):
            continue

    dict_items: list[dict] = []
    for block in _extract_tolerant_dict_blocks(text):
        try:
            item = _normalize_layout_item(ast.literal_eval(block), include_content)
            if item is not None:
                dict_items.append(item)
        except (SyntaxError, ValueError, TypeError, MemoryError, RecursionError):
            continue
    return dict_items if len(dict_items) > len(best) else best


def _map_bbox_to_image(bbox: list[float], width: int, height: int) -> list[float]:
    """Map MonkeyOCR's normalized 0–1000 coordinates to page pixels."""
    x1, y1, x2, y2 = bbox
    x1, x2 = x1 / 1000.0 * width, x2 / 1000.0 * width
    y1, y2 = y1 / 1000.0 * height, y2 / 1000.0 * height
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    x1 = max(0, min(round(x1), width - 1 if width > 0 else 0))
    y1 = max(0, min(round(y1), height - 1 if height > 0 else 0))
    x2 = max(x1 + 1, min(round(x2), width))
    y2 = max(y1 + 1, min(round(y2), height))
    return [float(x1), float(y1), float(x2), float(y2)]


def _parse_layout(
    content: str,
    image_size: tuple[int, int] = (1000, 1000),
    *,
    include_content: bool = True,
) -> list[LayoutElement]:
    """Parse MonkeyOCR output using the official demo's tolerant strategy."""
    width, height = image_size
    elements: list[LayoutElement] = []
    seen: set[tuple[tuple[float, ...], str]] = set()
    for item in _parse_tolerant_items(content, include_content):
        coords = _map_bbox_to_image(item["bbox"], width, height)
        identity = (tuple(coords), item["label"])
        if identity in seen:
            continue
        seen.add(identity)
        text = item.get("content", "")
        label = item["label"]
        included_in_markdown = KEEP_HEADER_FOOTER or _normalized_label(label) not in {
            "page-header",
            "page-footer",
        }
        elements.append(
            LayoutElement(
                bbox=coords,
                label=label,
                content=text,
                included_in_markdown=included_in_markdown,
            )
        )
    return elements


def _otsl_to_html(otsl: str) -> str:
    """Convert MonkeyOCR's OTSL table representation to previewable HTML."""
    if not otsl or not otsl.strip():
        return "<table></table>"

    rows = otsl.split("<nl>")
    if rows and not rows[-1]:
        rows.pop()
    grid: list[list[dict | None]] = []

    for row_index, row in enumerate(rows):
        if row_index >= len(grid):
            grid.append([])
        if not row.strip():
            continue

        column = 0
        for tag, content in re.findall(
            r"<([a-z]+)>(.*?)(?=<[a-z]+>|$)", row, flags=re.DOTALL
        ):
            while True:
                while len(grid[row_index]) <= column:
                    grid[row_index].append(None)
                if grid[row_index][column] is None:
                    break
                column += 1

            if tag in {"fcel", "ecel"}:
                grid[row_index][column] = {
                    "text": content.strip() if tag == "fcel" else "",
                    "rowspan": 1,
                    "colspan": 1,
                    "valid": True,
                }
            elif tag == "lcel":
                owner = next(
                    (
                        grid[row_index][candidate]
                        for candidate in range(column - 1, -1, -1)
                        if grid[row_index][candidate]
                        and grid[row_index][candidate].get("valid")
                    ),
                    None,
                )
                if owner:
                    owner["colspan"] += 1
                    grid[row_index][column] = {"valid": False}
                else:
                    grid[row_index][column] = {
                        "text": "",
                        "rowspan": 1,
                        "colspan": 1,
                        "valid": True,
                    }
            elif tag == "ucel":
                owner = next(
                    (
                        grid[candidate][column]
                        for candidate in range(row_index - 1, -1, -1)
                        if len(grid[candidate]) > column
                        and grid[candidate][column]
                        and grid[candidate][column].get("valid")
                    ),
                    None,
                )
                if owner:
                    owner["rowspan"] += 1
                    grid[row_index][column] = {"valid": False}
                else:
                    grid[row_index][column] = {
                        "text": "",
                        "rowspan": 1,
                        "colspan": 1,
                        "valid": True,
                    }
            elif tag == "xcel":
                grid[row_index][column] = {"valid": False}
            column += 1

    html = ["<table>"]
    for row in grid:
        html.append("<tr>")
        for cell in row:
            if not cell or not cell.get("valid"):
                continue
            attributes = []
            if cell["rowspan"] > 1:
                attributes.append(f'rowspan="{cell["rowspan"]}"')
            if cell["colspan"] > 1:
                attributes.append(f'colspan="{cell["colspan"]}"')
            attribute_text = " " + " ".join(attributes) if attributes else ""
            cell_text = "<br>".join(
                escape(part)
                for part in str(cell["text"])
                .replace("\r\n", "\n")
                .replace("\r", "\n")
                .split("\n")
            )
            html.append(f"<td{attribute_text}>{cell_text}</td>")
        html.append("</tr>")
    html.append("</table>")
    return "".join(html)


def _process_formula(content: str) -> str:
    content = content.strip("$").strip()
    content = re.sub(r"(?:\\quad\s*){5,}", r"\\quad ", content)
    content = re.sub(r"(?:\\qquad\s*){5,}", r"\\qquad ", content).strip()

    equation_number = None
    number_pattern = (
        r"(?:\\quad|\\qquad|\\eqno)\s*\(([^()]*)\)\s*$|\\tag\{([^{}]*)\}\s*$"
    )
    match = re.search(number_pattern, content)
    if match:
        equation_number = match.group(1) or match.group(2)
        content = content[: match.start()].rstrip()

    begin_environment = None
    begin_match = re.match(r"^\\begin\{([^}]+)\}", content)
    if begin_match:
        begin_environment = begin_match.group(1)
        content = content[begin_match.end() :].lstrip()
        end_match = re.search(
            rf"\\end\{{{re.escape(begin_environment)}\}}\s*$", content
        )
        if end_match:
            content = content[: end_match.start()].rstrip()

    match = re.search(number_pattern, content)
    if match:
        equation_number = match.group(1) or match.group(2)
        content = content[: match.start()].rstrip()
    if begin_environment:
        content = (
            f"\\begin{{{begin_environment}}}\n{content}\n\\end{{{begin_environment}}}"
        )

    rendered = f"$$\n{content}\n$$"
    return f"{rendered}\n{equation_number}" if equation_number else rendered


TABLE_IMAGE_PATTERN = re.compile(
    r"\[img\]\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*"
    r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*"
    r"(-?\d+(?:\.\d+)?)\s*\]\s*\[/img\]",
    flags=re.IGNORECASE,
)


def _replace_table_images(content: str, table_image: Image.Image | None) -> str:
    if table_image is None or not TABLE_IMAGE_PATTERN.search(content):
        return content
    references: dict[tuple[float, ...], str] = {}

    def replace(match: re.Match) -> str:
        normalized_bbox = tuple(float(value) for value in match.groups())
        image_uri = references.get(normalized_bbox)
        if image_uri is None:
            bbox = _map_bbox_to_image(list(normalized_bbox), *table_image.size)
            embedded = table_image.crop(tuple(round(value) for value in bbox))
            try:
                image_uri = _image_data_uri(embedded)
            finally:
                embedded.close()
            references[normalized_bbox] = image_uri
        return (
            f'<img src="{escape(image_uri, quote=True)}" alt="embedded table image" />'
        )

    return TABLE_IMAGE_PATTERN.sub(replace, content)


def _normalized_label(label: str) -> str:
    return re.sub(r"[\s_]+", "-", label.strip().lower())


def _format_element_content(
    element: LayoutElement, crop: Image.Image | None = None
) -> str:
    content = element.content.strip()
    label = _normalized_label(element.label)
    if label == "formula":
        return content if content.startswith("$$") else _process_formula(content)
    if label == "table":
        table = (
            content
            if content.lstrip().lower().startswith("<table")
            else _otsl_to_html(content)
        )
        return _replace_table_images(table, crop)
    if label in {"picture", "image", "figure"} and crop is not None:
        return f"![image]({_image_data_uri(crop)})"
    if label in {"title", "document-title"}:
        return (
            content
            if content.startswith("# ")
            else "# " + content.replace("\n", "\n# ")
        )
    if label in {"section-header", "heading", "header"}:
        return (
            content
            if content.startswith("## ")
            else "## " + content.replace("\n", "\n## ")
        )
    return content


def _elements_to_markdown(elements: list[LayoutElement], fallback: str) -> str:
    if not elements:
        return fallback.strip()

    blocks: list[str] = []
    for element in elements:
        if not element.included_in_markdown or (
            not KEEP_HEADER_FOOTER
            and _normalized_label(element.label) in {"page-header", "page-footer"}
        ):
            continue
        content = _format_element_content(element)
        if not content:
            continue
        blocks.append(content)
    return "\n\n".join(blocks).replace("�", "").strip()


async def _read_upload(file: UploadFile) -> tuple[bytes, str]:
    content_type = (file.content_type or "").lower()
    if content_type not in ALLOWED_TYPES:
        await file.close()
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type. Upload a PNG, JPEG, WebP, BMP, TIFF, or PDF.",
        )
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    await file.close()
    if not data:
        raise HTTPException(status_code=422, detail="The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"The upload exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.",
        )
    return data, content_type


def _detect_repeat_token(
    output: str,
    base_max_repeats: int = 4,
    window_size: int = 500,
    cut_from_end: int = 0,
    scaling_factor: float = 3.0,
) -> bool:
    """Detect the repeated suffix failure mode produced by MonkeyOCR."""
    if cut_from_end > 0:
        output = output[:-cut_from_end]
    if not output:
        return False

    for sequence_length in range(1, min(window_size // 2, len(output)) + 1):
        sequence = output[-sequence_length:]
        max_repeats = int(base_max_repeats * (1 + scaling_factor / sequence_length))
        if output.endswith(sequence * (max_repeats + 1)):
            return True
    return False


def _should_retry_repeated_output(output: str) -> bool:
    return _detect_repeat_token(output) or (
        len(output) > 50 and _detect_repeat_token(output, cut_from_end=50)
    )


async def _post_ocr_payload(
    client: httpx.AsyncClient,
    payload: dict,
    page_number: int,
    semaphore: asyncio.Semaphore,
) -> str:
    last_error: Exception | None = None
    for attempt in range(HTTP_MAX_RETRIES + 1):
        try:
            async with semaphore:
                response = await client.post("/chat/completions", json=payload)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise ValueError("empty model response")
            return content
        except (
            httpx.TimeoutException,
            httpx.NetworkError,
            httpx.HTTPStatusError,
        ) as exc:
            last_error = exc
            retryable = not isinstance(
                exc, httpx.HTTPStatusError
            ) or exc.response.status_code in {429, 500, 502, 503, 504}
            if not retryable or attempt == HTTP_MAX_RETRIES:
                break
            await asyncio.sleep(min(HTTP_RETRY_BACKOFF * (2**attempt), 30.0))
        except (KeyError, TypeError, ValueError) as exc:
            last_error = exc
            break

    detail = f"OCR failed on page {page_number}."
    if isinstance(last_error, httpx.HTTPStatusError):
        detail += f" Upstream returned HTTP {last_error.response.status_code}."
    elif isinstance(last_error, (httpx.TimeoutException, httpx.NetworkError)):
        detail += " The OCR service is unavailable or timed out."
    else:
        detail += " The OCR service returned an unexpected response."
    raise HTTPException(status_code=502, detail=detail)


async def _request_ocr(
    client: httpx.AsyncClient,
    image: Image.Image,
    prompt: str,
    page_number: int,
    semaphore: asyncio.Semaphore,
    *,
    max_tokens: int | None = None,
    retry_repetition: bool = True,
    min_pixels: int = 0,
) -> str:
    prepared_image = _prepare_model_image(image, min_pixels=min_pixels)
    try:
        image_url = _image_data_uri(prepared_image)
    finally:
        if prepared_image is not image:
            prepared_image.close()

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    content = ""
    repeat_attempts = REPEAT_MAX_RETRIES if retry_repetition else 0
    for attempt in range(repeat_attempts + 1):
        request_payload = {
            **payload,
            "temperature": 0 if attempt == 0 else min(0.2 * attempt, 0.8),
        }
        if attempt > 0:
            request_payload["top_p"] = 0.95
        content = await _post_ocr_payload(
            client,
            request_payload,
            page_number,
            semaphore,
        )
        if not _should_retry_repeated_output(content):
            return content
    return content


def _crop_element(image: Image.Image, element: LayoutElement) -> Image.Image:
    return image.crop(tuple(round(value) for value in element.bbox))


async def _recognize_layout_elements(
    client: httpx.AsyncClient,
    image: Image.Image,
    elements: list[LayoutElement],
    page_number: int,
    semaphore: asyncio.Semaphore,
) -> list[LayoutElement]:
    async def recognize(element: LayoutElement) -> LayoutElement:
        crop = _crop_element(image, element)
        try:
            prompt = MONKEYOCR_PROMPTS.get(element.label)
            if prompt:
                raw_content = await _request_ocr(
                    client,
                    crop,
                    prompt,
                    page_number,
                    semaphore,
                    max_tokens=4096 if element.label == "Table" else 5000,
                )
                source = LayoutElement(
                    bbox=element.bbox,
                    label=element.label,
                    content=raw_content,
                    included_in_markdown=element.included_in_markdown,
                )
                content = _format_element_content(source, crop)
            elif _normalized_label(element.label) in {"picture", "image", "figure"}:
                content = _format_element_content(element, crop)
            else:
                content = ""
            return LayoutElement(
                bbox=element.bbox,
                label=element.label,
                content=content,
                included_in_markdown=element.included_in_markdown,
            )
        finally:
            crop.close()

    return list(await asyncio.gather(*(recognize(element) for element in elements)))


async def _ocr_page(
    client: httpx.AsyncClient,
    image: Image.Image,
    page_number: int,
    semaphore: asyncio.Semaphore,
) -> tuple[str, list[LayoutElement]]:
    if PIPELINE_MODE == "end2end":
        raw = await _request_ocr(
            client,
            image,
            END2END_PROMPT,
            page_number,
            semaphore,
            max_tokens=8192,
        )
        elements = _parse_layout(raw, image.size)
        formatted: list[LayoutElement] = []
        for element in elements:
            crop = _crop_element(image, element)
            try:
                formatted.append(
                    LayoutElement(
                        bbox=element.bbox,
                        label=element.label,
                        content=_format_element_content(element, crop),
                        included_in_markdown=element.included_in_markdown,
                    )
                )
            finally:
                crop.close()
        return raw, formatted

    raw = await _request_ocr(
        client,
        image,
        MONKEYOCR_PROMPTS["LAYOUT"],
        page_number,
        semaphore,
        max_tokens=4096,
        retry_repetition=False,
        min_pixels=MIN_IMAGE_PIXELS,
    )
    layout = _parse_layout(raw, image.size, include_content=False)
    return raw, await _recognize_layout_elements(
        client, image, layout, page_number, semaphore
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ocr_client = httpx.AsyncClient(
        base_url=API_BASE_URL,
        timeout=httpx.Timeout(REQUEST_TIMEOUT),
        limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        headers={
            "Authorization": f"Bearer {os.getenv('MONKEYOCR_API_KEY', 'not-required')}"
        },
    )
    yield
    await app.state.ocr_client.aclose()


app = FastAPI(
    title="ParseAnything",
    version="1.2.0",
    description="Image and PDF parsing backed by MonkeyOCRv2.",
    lifespan=lifespan,
)


@app.get("/api/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    try:
        response = await request.app.state.ocr_client.get("/models", timeout=10)
        response.raise_for_status()
        models = {item.get("id") for item in response.json().get("data", [])}
        if MODEL not in models:
            return HealthResponse(
                status="degraded",
                model=MODEL,
                upstream="model unavailable",
                pipeline=PIPELINE_MODE,
            )
        return HealthResponse(
            status="ok", model=MODEL, upstream="connected", pipeline=PIPELINE_MODE
        )
    except Exception:
        return HealthResponse(
            status="degraded",
            model=MODEL,
            upstream="unreachable",
            pipeline=PIPELINE_MODE,
        )


@app.post("/api/preview", response_model=PreviewResponse)
async def preview_document(file: UploadFile = File(...)) -> PreviewResponse:
    data, content_type = await _read_upload(file)
    filename = Path(file.filename or "document").name
    if content_type == PDF_TYPE:
        page_count = await asyncio.to_thread(_pdf_page_count, data)
        rendered = await asyncio.to_thread(
            _render_pdf_pages,
            data,
            list(range(1, page_count + 1)),
            MAX_PREVIEW_SIDE,
            0,
        )
    else:
        page_count = await asyncio.to_thread(_image_page_count, data)
        rendered = await asyncio.to_thread(
            _render_image_pages,
            data,
            list(range(1, page_count + 1)),
            MAX_PREVIEW_SIDE,
            0,
        )

    try:
        pages = [
            PagePreview(page_number=number, image_url=_preview_data_uri(image))
            for number, image in rendered
        ]
    finally:
        for _, image in rendered:
            image.close()
    return PreviewResponse(filename=filename, page_count=page_count, pages=pages)


@app.post("/api/parse", response_model=ParseResponse)
async def parse_document(
    request: Request,
    file: UploadFile = File(...),
    selected_pages: str | None = Form(default=None),
    batch_size: int = Form(default=1),
) -> ParseResponse:
    data, content_type = await _read_upload(file)
    filename = Path(file.filename or "document").name

    if content_type == PDF_TYPE:
        source_page_count = await asyncio.to_thread(_pdf_page_count, data)
        page_numbers = _parse_page_selection(selected_pages, source_page_count)
        batch_size = _validate_batch_size(batch_size, len(page_numbers))
        rendered = await asyncio.to_thread(
            _render_pdf_pages,
            data,
            page_numbers,
            MAX_IMAGE_SIDE,
            MIN_IMAGE_PIXELS,
        )
    else:
        source_page_count = await asyncio.to_thread(_image_page_count, data)
        page_numbers = _parse_page_selection(selected_pages, source_page_count)
        batch_size = _validate_batch_size(batch_size, len(page_numbers))
        rendered = await asyncio.to_thread(
            _render_image_pages,
            data,
            page_numbers,
            MAX_IMAGE_SIDE,
            MIN_IMAGE_PIXELS,
        )

    started = perf_counter()
    request_semaphore = asyncio.Semaphore(max(1, OCR_CONCURRENCY))
    page_semaphore = asyncio.Semaphore(batch_size)

    async def parse_page(page_number: int, image: Image.Image) -> PageResult:
        async with page_semaphore:
            try:
                (raw, elements), image_url = await asyncio.gather(
                    _ocr_page(
                        request.app.state.ocr_client,
                        image,
                        page_number,
                        request_semaphore,
                    ),
                    asyncio.to_thread(_preview_data_uri, image),
                )
                return PageResult(
                    page_number=page_number,
                    markdown=_elements_to_markdown(
                        elements, raw if PIPELINE_MODE == "end2end" else ""
                    ),
                    raw=raw,
                    elements=elements,
                    image_url=image_url,
                    image_width=image.width,
                    image_height=image.height,
                )
            finally:
                image.close()

    results = await asyncio.gather(
        *(parse_page(number, image) for number, image in rendered)
    )
    markdown_pages = [result.markdown for result in results]
    combined = (
        markdown_pages[0]
        if len(results) == 1
        else "\n\n".join(
            f"<!-- Page {result.page_number} -->\n\n{result.markdown}"
            for result in results
        )
    )

    return ParseResponse(
        filename=filename,
        page_count=len(results),
        source_page_count=source_page_count,
        selected_pages=page_numbers,
        batch_size=batch_size,
        elapsed_seconds=round(perf_counter() - started, 2),
        content=combined,
        pages=markdown_pages,
        page_results=results,
    )


@app.get("/sample")
async def sample() -> Response:
    if SAMPLE_IMAGE.exists():
        return FileResponse(
            SAMPLE_IMAGE, media_type="image/png", filename="page-62.png"
        )

    image = Image.new("RGB", (1000, 1300), "white")
    draw = ImageDraw.Draw(image)
    try:
        title_font = ImageFont.truetype("DejaVuSans.ttf", 42)
        body_font = ImageFont.truetype("DejaVuSans.ttf", 28)
    except OSError:
        title_font = body_font = ImageFont.load_default()
    draw.text((90, 100), "MONKEYOCR SAMPLE", fill="#186f4b", font=body_font)
    draw.text(
        (90, 175), "Document intelligence report", fill="#151714", font=title_font
    )
    draw.line((90, 225, 910, 225), fill="#d8d7d0", width=3)
    draw.text((90, 285), "Revenue", fill="#151714", font=body_font)
    draw.text((720, 285), "$12,500", fill="#151714", font=body_font)
    draw.text(
        (90, 350),
        "Net income increased by 18% this quarter.",
        fill="#151714",
        font=body_font,
    )
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    image.close()
    return Response(content=buffer.getvalue(), media_type="image/png")


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
