import base64
from types import SimpleNamespace

import fitz
from fastapi.testclient import TestClient

from app.main import app, parse_model_json


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)
TEMPLATE = b'{"name": "verbatim-string", "active": "boolean"}'


class FakeCompletions:
    def __init__(self) -> None:
        self.requests = []

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content='```json\n{"name": "Ada", "active": true}\n```'
                    )
                )
            ],
            usage=SimpleNamespace(completion_tokens=12),
        )


class FakeModels:
    async def list(self):
        return SimpleNamespace(data=[])


class FakeOpenAI:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=FakeCompletions())
        self.models = FakeModels()

    async def close(self) -> None:
        return None


def test_home_page_is_served() -> None:
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "NuExtract Studio" in response.text
    assert 'id="type-template"' in response.text
    assert 'id="edit-template"' in response.text
    assert 'id="template-editor-dialog"' in response.text
    assert "Key-value table" in response.text
    assert "Raw JSON" in response.text


def test_invalid_template_is_rejected_before_model_call() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/api/extract",
            files={
                "document": ("scan.png", PNG_1X1, "image/png"),
                "template": ("template.json", b"not json", "application/json"),
            },
        )

    assert response.status_code == 400
    assert "Invalid template JSON" in response.json()["detail"]


def test_image_extraction_returns_parsed_result() -> None:
    with TestClient(app) as client:
        fake_client = FakeOpenAI()
        app.state.nuextract_client = fake_client
        response = client.post(
            "/api/extract",
            files={
                "document": ("scan.png", PNG_1X1, "image/png"),
                "template": ("template.json", TEMPLATE, "application/json"),
            },
            data={"instructions": "Keep exact names", "thinking": "false"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["result"] == {"name": "Ada", "active": True}
    assert payload["metadata"]["pages"] == 1
    assert payload["metadata"]["completion_tokens"] == 12
    request = fake_client.chat.completions.requests[0]
    assert request["temperature"] == 0.2
    assert request["messages"][0]["content"][0]["type"] == "image_url"


def test_multi_page_pdf_is_rendered_in_page_order() -> None:
    document = fitz.open()
    document.new_page().insert_text((72, 72), "Page one")
    document.new_page().insert_text((72, 72), "Page two")
    pdf_bytes = document.tobytes()
    document.close()

    with TestClient(app) as client:
        fake_client = FakeOpenAI()
        app.state.nuextract_client = fake_client
        response = client.post(
            "/api/extract",
            files={
                "document": ("two-pages.pdf", pdf_bytes, "application/pdf"),
                "template": ("template.json", TEMPLATE, "application/json"),
            },
            data={"thinking": "true"},
        )

    assert response.status_code == 200
    assert response.json()["metadata"]["pages"] == 2
    request = fake_client.chat.completions.requests[0]
    assert request["temperature"] == 0.6
    assert len(request["messages"][0]["content"]) == 2


def test_reasoning_and_fenced_json_are_parsed() -> None:
    raw = '<think>Reasoning here</think>\n```json\n{"ok": true}\n```'
    assert parse_model_json(raw) == {"ok": True}
