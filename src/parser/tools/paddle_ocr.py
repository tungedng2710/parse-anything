"""Optional PaddleOCR adapter."""

from __future__ import annotations

from html import escape
from typing import Any

from src.parser.models import BoundingBox, ElementType, LayoutRegion, ParsedElement
from src.parser.tools_manager import OCRTool, OCRToolContext, OCRToolUnavailableError


class PaddleOCRTool(OCRTool):
    """Recognize text with PaddleOCR when the optional dependency is installed."""

    name = "paddle"
    capabilities = frozenset({"recognize_text"})

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._ocr: Any | None = None

    def recognize_text(self, context: OCRToolContext) -> ParsedElement:
        ocr = self._load_ocr()
        raw_result = ocr.ocr(context.page.image_path, cls=bool(self.config.get("use_angle_cls", True)))
        lines = self._extract_lines(raw_result)
        text = "\n".join(line["text"] for line in lines)
        confidence = min((line["confidence"] for line in lines), default=0.0)
        region = context.region or self._default_region(context)

        return ParsedElement(
            element_id=region.region_id,
            element_type=region.element_type,
            page_index=context.page.page_index,
            reading_order=region.reading_order,
            markdown=text,
            html=f"<p>{escape(text)}</p>" if text else "",
            text=text,
            data={"lines": lines},
            bbox=region.bbox,
            confidence=confidence,
            metadata={"processor": self.__class__.__name__, "image_path": context.page.image_path, "tool": self.name},
        )

    def _load_ocr(self) -> Any:
        if self._ocr is None:
            try:
                from paddleocr import PaddleOCR
            except ImportError as exc:  # pragma: no cover - optional dependency.
                raise OCRToolUnavailableError("PaddleOCR is not installed") from exc

            self._ocr = PaddleOCR(
                lang=str(self.config.get("language", "en")),
                use_angle_cls=bool(self.config.get("use_angle_cls", True)),
            )
        return self._ocr

    def _extract_lines(self, raw_result: Any) -> list[dict[str, Any]]:
        lines: list[dict[str, Any]] = []
        for page_result in raw_result or []:
            for item in page_result or []:
                if len(item) < 2:
                    continue
                bbox, recognition = item[0], item[1]
                if not isinstance(recognition, (list, tuple)) or len(recognition) < 2:
                    continue
                lines.append({"bbox": bbox, "text": str(recognition[0]), "confidence": float(recognition[1])})
        return lines

    def _default_region(self, context: OCRToolContext) -> LayoutRegion:
        return LayoutRegion(
            region_id=f"p{context.page.page_index}-text-1",
            element_type=ElementType.TEXT,
            bbox=BoundingBox(x=0.0, y=0.0, width=1.0, height=1.0),
            reading_order=1,
            metadata={"image_path": context.page.image_path, "tool": self.name},
        )
