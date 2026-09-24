"""Optional Chandra OCR adapter.

This module intentionally avoids loading the model at import time. The model is
large, so it is initialized only when the tool is selected and called.
"""

from __future__ import annotations

from html import escape
from typing import Any

from src.parser.models import BoundingBox, ElementType, LayoutRegion, PageLayoutGraph, ParsedElement
from src.parser.tools_manager import OCRTool, OCRToolContext, OCRToolUnavailableError


class ChandraOCRTool(OCRTool):
    """Use Chandra OCR for page-level markdown OCR when dependencies exist."""

    name = "chandra"
    capabilities = frozenset({"analyze_layout", "recognize_text"})

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._model: Any | None = None

    def analyze_layout(self, context: OCRToolContext) -> PageLayoutGraph:
        region = LayoutRegion(
            region_id=f"p{context.page.page_index}-text-1",
            element_type=ElementType.TEXT,
            bbox=BoundingBox(x=0.0, y=0.0, width=1.0, height=1.0),
            reading_order=1,
            metadata={"image_path": context.page.image_path, "tool": self.name},
        )
        return PageLayoutGraph(session_id=context.session_id, page_index=context.page.page_index, regions=[region])

    def recognize_text(self, context: OCRToolContext) -> ParsedElement:
        markdown = self._generate_markdown(context.page.image_path)
        text = markdown.strip()
        region = context.region or LayoutRegion(
            region_id=f"p{context.page.page_index}-text-1",
            element_type=ElementType.TEXT,
            bbox=BoundingBox(x=0.0, y=0.0, width=1.0, height=1.0),
            reading_order=1,
            metadata={"image_path": context.page.image_path, "tool": self.name},
        )

        return ParsedElement(
            element_id=region.region_id,
            element_type=region.element_type,
            page_index=context.page.page_index,
            reading_order=region.reading_order,
            markdown=markdown,
            html=f"<pre>{escape(markdown)}</pre>" if markdown else "",
            text=text,
            bbox=region.bbox,
            confidence=region.confidence,
            metadata={"processor": self.__class__.__name__, "image_path": context.page.image_path, "tool": self.name},
        )

    def _generate_markdown(self, image_path: str) -> str:
        model = self._load_model()
        try:
            from chandra.model.hf import generate_hf
            from chandra.model.schema import BatchInputItem
            from chandra.output import parse_markdown
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - optional dependency.
            raise OCRToolUnavailableError("Chandra OCR dependencies are not installed") from exc

        batch = [
            BatchInputItem(
                image=Image.open(image_path),
                prompt_type=str(self.config.get("prompt_type", "ocr_layout")),
            )
        ]
        result = generate_hf(batch, model)[0]
        return str(parse_markdown(result.raw))

    def _load_model(self) -> Any:
        if self._model is None:
            try:
                import torch
                from transformers import AutoModelForImageTextToText, AutoProcessor
            except ImportError as exc:  # pragma: no cover - optional dependency.
                raise OCRToolUnavailableError("Chandra OCR model dependencies are not installed") from exc

            model_name = str(self.config.get("model_name", "datalab-to/chandra-ocr-2"))
            model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                dtype=torch.bfloat16,
                device_map="auto",
            )
            model.eval()
            model.processor = AutoProcessor.from_pretrained(model_name)
            model.processor.tokenizer.padding_side = "left"
            self._model = model
        return self._model
