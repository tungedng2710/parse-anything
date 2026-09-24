import asyncio
import base64
import json
from io import BytesIO

import httpx
from fastapi.testclient import TestClient
from PIL import Image

from backend.main import (
    MAX_IMAGE_SIDE,
    MIN_IMAGE_PIXELS,
    MODEL_MAX_PIXELS,
    _detect_repeat_token,
    _elements_to_markdown,
    _map_bbox_to_image,
    _otsl_to_html,
    _parse_layout,
    _parse_page_selection,
    _process_formula,
    _prepare_model_image,
    _normalize_image,
    _replace_table_images,
    app,
)


def _png() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (120, 80), "white").save(buffer, "PNG")
    return buffer.getvalue()


def _pdf(page_count: int = 1) -> bytes:
    buffer = BytesIO()
    pages = [Image.new("RGB", (120, 80), "white") for _ in range(page_count)]
    pages[0].save(buffer, "PDF", save_all=True, append_images=pages[1:])
    return buffer.getvalue()


def _tiff(page_count: int = 2) -> bytes:
    buffer = BytesIO()
    pages = [Image.new("RGB", (120, 80), "white") for _ in range(page_count)]
    pages[0].save(buffer, "TIFF", save_all=True, append_images=pages[1:])
    return buffer.getvalue()


def test_parse_image():
    recognition_outputs = iter(["Parsed", "Hello"])

    async def upstream(request: httpx.Request) -> httpx.Response:
        prompt = json.loads(request.content)["messages"][0]["content"][1]["text"]
        if "categories and coordinates" in prompt:
            content = (
                "[{'bbox': [10, 20, 400, 80], 'label': 'Title'}, "
                "{'bbox': [10, 100, 400, 150], 'label': 'Text'}]"
            )
        else:
            content = next(recognition_outputs)
        return httpx.Response(
            200, json={"choices": [{"message": {"content": content}}]}
        )

    with TestClient(app) as client:
        real_client = app.state.ocr_client
        mock_client = httpx.AsyncClient(
            transport=httpx.MockTransport(upstream), base_url="https://ocr.test/v1"
        )
        app.state.ocr_client = mock_client
        try:
            response = client.post(
                "/api/parse", files={"file": ("page.png", _png(), "image/png")}
            )
        finally:
            app.state.ocr_client = real_client

    assert response.status_code == 200
    assert response.json()["content"] == "# Parsed\n\nHello"
    assert response.json()["page_count"] == 1
    assert response.json()["selected_pages"] == [1]
    page = response.json()["page_results"][0]
    assert page["image_width"] * page["image_height"] >= MIN_IMAGE_PIXELS
    assert max(page["image_width"], page["image_height"]) <= MAX_IMAGE_SIDE
    assert page["elements"][0]["bbox"] == _map_bbox_to_image(
        [10, 20, 400, 80], page["image_width"], page["image_height"]
    )
    assert page["image_url"].startswith("data:image/jpeg;base64,")


def test_repeated_recognition_output_is_retried():
    recognition_temperatures = []

    async def upstream(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        prompt = payload["messages"][0]["content"][1]["text"]
        if "categories and coordinates" in prompt:
            content = "[{'bbox': [0, 0, 1000, 1000], 'label': 'Text'}]"
        else:
            recognition_temperatures.append(payload["temperature"])
            content = (
                "10\n" * 200 if len(recognition_temperatures) == 1 else "Recovered text"
            )
        return httpx.Response(
            200, json={"choices": [{"message": {"content": content}}]}
        )

    with TestClient(app) as client:
        real_client = app.state.ocr_client
        app.state.ocr_client = httpx.AsyncClient(
            transport=httpx.MockTransport(upstream), base_url="https://ocr.test/v1"
        )
        try:
            response = client.post(
                "/api/parse", files={"file": ("page.png", _png(), "image/png")}
            )
        finally:
            app.state.ocr_client = real_client

    assert response.status_code == 200
    assert response.json()["content"] == "Recovered text"
    assert recognition_temperatures == [0, 0.2]
    assert _detect_repeat_token("10\n" * 200)


def test_page_batch_size_limits_parallel_ocr():
    active = 0
    peak = 0

    async def upstream(request: httpx.Request) -> httpx.Response:
        nonlocal active, peak
        prompt = json.loads(request.content)["messages"][0]["content"][1]["text"]
        is_layout = "categories and coordinates" in prompt
        if is_layout:
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1
        content = (
            "[{'bbox': [0, 0, 1000, 1000], 'label': 'Text'}]"
            if is_layout
            else "Recognized"
        )
        return httpx.Response(
            200, json={"choices": [{"message": {"content": content}}]}
        )

    with TestClient(app) as client:
        real_client = app.state.ocr_client
        app.state.ocr_client = httpx.AsyncClient(
            transport=httpx.MockTransport(upstream), base_url="https://ocr.test/v1"
        )
        try:
            response = client.post(
                "/api/parse",
                files={"file": ("document.pdf", _pdf(3), "application/pdf")},
                data={"batch_size": "2"},
            )
        finally:
            app.state.ocr_client = real_client

    assert peak == 2
    assert response.status_code == 200
    assert response.json()["batch_size"] == 2


def test_rejects_unsupported_type():
    with TestClient(app) as client:
        response = client.post(
            "/api/parse", files={"file": ("notes.txt", b"hello", "text/plain")}
        )
    assert response.status_code == 415


def test_parse_pdf():
    calls = 0
    layout_image_sizes = []

    async def upstream(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        prompt = json.loads(request.content)["messages"][0]["content"][1]["text"]
        if "categories and coordinates" in prompt:
            image_url = json.loads(request.content)["messages"][0]["content"][0][
                "image_url"
            ]["url"]
            image_data = base64.b64decode(image_url.split(",", 1)[1])
            with Image.open(BytesIO(image_data)) as rasterized_page:
                assert rasterized_page.format == "PNG"
                layout_image_sizes.append(rasterized_page.size)
        content = (
            "[{'bbox': [0, 0, 1000, 1000], 'label': 'Text'}]"
            if "categories and coordinates" in prompt
            else "PDF page"
        )
        return httpx.Response(
            200, json={"choices": [{"message": {"content": content}}]}
        )

    with TestClient(app) as client:
        real_client = app.state.ocr_client
        app.state.ocr_client = httpx.AsyncClient(
            transport=httpx.MockTransport(upstream), base_url="https://ocr.test/v1"
        )
        try:
            response = client.post(
                "/api/parse",
                files={"file": ("document.pdf", _pdf(3), "application/pdf")},
                data={"selected_pages": "2"},
            )
        finally:
            app.state.ocr_client = real_client

    assert response.status_code == 200
    assert response.json()["content"] == "PDF page"
    assert response.json()["page_count"] == 1
    assert response.json()["source_page_count"] == 3
    assert response.json()["selected_pages"] == [2]
    assert response.json()["page_results"][0]["page_number"] == 2
    assert calls == 2
    assert len(layout_image_sizes) == 1
    layout_pixels = layout_image_sizes[0][0] * layout_image_sizes[0][1]
    assert abs(layout_pixels - MODEL_MAX_PIXELS) / MODEL_MAX_PIXELS < 0.01


def test_preview_pdf_pages():
    with TestClient(app) as client:
        response = client.post(
            "/api/preview", files={"file": ("document.pdf", _pdf(2), "application/pdf")}
        )

    assert response.status_code == 200
    assert response.json()["page_count"] == 2
    assert [page["page_number"] for page in response.json()["pages"]] == [1, 2]
    assert all(
        page["image_url"].startswith("data:image/jpeg;base64,")
        for page in response.json()["pages"]
    )


def test_parse_selected_tiff_frame():
    calls = 0

    async def upstream(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        prompt = json.loads(request.content)["messages"][0]["content"][1]["text"]
        content = (
            "[{'bbox': [0, 0, 1000, 1000], 'label': 'Text'}]"
            if "categories and coordinates" in prompt
            else "TIFF frame"
        )
        return httpx.Response(
            200, json={"choices": [{"message": {"content": content}}]}
        )

    with TestClient(app) as client:
        real_client = app.state.ocr_client
        app.state.ocr_client = httpx.AsyncClient(
            transport=httpx.MockTransport(upstream), base_url="https://ocr.test/v1"
        )
        try:
            response = client.post(
                "/api/parse",
                files={"file": ("scan.tiff", _tiff(3), "image/tiff")},
                data={"selected_pages": "2"},
            )
        finally:
            app.state.ocr_client = real_client

    assert response.status_code == 200
    assert response.json()["source_page_count"] == 3
    assert response.json()["page_results"][0]["page_number"] == 2
    assert calls == 2


def test_generated_sample_is_available():
    with TestClient(app) as client:
        response = client.get("/sample")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content.startswith(b"\x89PNG")


def test_page_selection_ranges_and_validation():
    assert _parse_page_selection("1,3-5,3", 5) == [1, 3, 4, 5]

    with TestClient(app) as client:
        response = client.post(
            "/api/parse",
            files={"file": ("document.pdf", _pdf(2), "application/pdf")},
            data={"selected_pages": "3"},
        )
    assert response.status_code == 422
    assert "outside" in response.json()["detail"]

    with TestClient(app) as client:
        response = client.post(
            "/api/parse",
            files={"file": ("document.pdf", _pdf(2), "application/pdf")},
            data={"selected_pages": "1-2", "batch_size": "3"},
        )
    assert response.status_code == 422
    assert "between 1 and 2" in response.json()["detail"]


def test_monkey_output_parser_supports_code_fenced_python():
    raw = """```python
    [{'bbox': [1, 2, 300, 80], 'label': 'Section-header', 'content': 'Summary'},
     {'bbox': [1, 90, 600, 180], 'label': 'Text', 'content': 'Result text'}]
    ```"""
    elements = _parse_layout(raw)

    assert len(elements) == 2
    assert _elements_to_markdown(elements, raw) == "## Summary\n\nResult text"


def test_small_images_are_upscaled_for_ocr_without_exceeding_max_side():
    source = Image.new("RGB", (120, 80), "white")
    result = _normalize_image(source, max_side=500, min_pixels=100_000)
    try:
        assert result.width * result.height >= 100_000
        assert max(result.size) <= 500
        assert abs(result.width / result.height - 1.5) < 0.01
    finally:
        result.close()
        source.close()


def test_model_images_use_the_official_pixel_budget():
    source = Image.new("RGB", (1700, 2200), "white")
    result = _prepare_model_image(
        source,
        min_pixels=MODEL_MAX_PIXELS,
        max_pixels=MODEL_MAX_PIXELS,
    )
    try:
        assert (
            abs(result.width * result.height - MODEL_MAX_PIXELS) / MODEL_MAX_PIXELS
            < 0.01
        )
        assert result.size != source.size
        assert abs(result.width / result.height - source.width / source.height) < 0.01
    finally:
        result.close()
        source.close()


def test_monkey_output_parser_recovers_items_and_maps_pixel_boxes():
    raw = """Layout follows:
    {'bbox': [800, 900, 100, 200], 'label': 'Text'},
    {'bbox': [-20, -20, 1200, 1200], 'label': 'Picture'}
    """
    elements = _parse_layout(raw, (200, 100), include_content=False)

    assert [element.bbox for element in elements] == [
        [20.0, 20.0, 160.0, 90.0],
        [0.0, 0.0, 200.0, 100.0],
    ]
    assert _map_bbox_to_image([500, 500, 500, 500], 100, 100) == [
        50.0,
        50.0,
        51.0,
        51.0,
    ]


def test_layout_parser_deduplicates_repeated_boxes():
    repeated = (
        "[{'bbox': [10, 20, 300, 80], 'label': 'Text'}, "
        "{'bbox': [10, 20, 300, 80], 'label': 'Text'}]"
    )

    assert len(_parse_layout(repeated)) == 1


def test_layout_elements_expose_markdown_inclusion():
    raw = (
        "[{'bbox': [0, 0, 1000, 100], 'label': 'Page-header', "
        "'content': 'Header'}, {'bbox': [0, 100, 1000, 900], "
        "'label': 'Text', 'content': 'Body'}]"
    )

    elements = _parse_layout(raw)

    assert [element.included_in_markdown for element in elements] == [False, True]
    assert _elements_to_markdown(elements, raw) == "Body"


def test_otsl_tables_are_rendered_as_html():
    assert _otsl_to_html("<fcel>Name<fcel>Value<nl><fcel>A<fcel>1") == (
        "<table><tr><td>Name</td><td>Value</td></tr>"
        "<tr><td>A</td><td>1</td></tr></table>"
    )


def test_formula_formatting_matches_official_markdown():
    assert _process_formula(r"\begin{aligned}x &= 1\end{aligned}\tag{4}") == (
        "$$\n\\begin{aligned}\nx &= 1\n\\end{aligned}\n$$\n4"
    )


def test_table_image_markers_embed_the_selected_crop():
    image = Image.new("RGB", (100, 80), "white")
    try:
        result = _replace_table_images(
            "<table><tr><td>[img][0, 0, 500, 500][/img]</td></tr></table>",
            image,
        )
    finally:
        image.close()

    assert '<img src="data:image/png;base64,' in result
    assert 'alt="embedded table image"' in result
