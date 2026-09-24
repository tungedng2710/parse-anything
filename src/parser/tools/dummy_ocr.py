"""Deterministic OCR backend used by tests and local development."""

from __future__ import annotations

from html import escape

from src.parser.models import BoundingBox, ElementType, LayoutRegion, PageLayoutGraph, ParsedElement
from src.parser.tools_manager import OCRTool, OCRToolContext


class DummyOCRTool(OCRTool):
    """Return stable parser objects without calling an external OCR service."""

    name = "dummy"
    capabilities = frozenset({"analyze_layout", "detect_table", "detect_figure", "recognize_text"})

    def analyze_layout(self, context: OCRToolContext) -> PageLayoutGraph:
        page = context.page
        regions = [
            LayoutRegion(
                region_id=f"p{page.page_index}-text-1",
                element_type=ElementType.TEXT,
                bbox=BoundingBox(x=0.05, y=0.08, width=0.9, height=0.2),
                reading_order=1,
                metadata={"image_path": page.image_path, "tool": self.name},
            ),
            LayoutRegion(
                region_id=f"p{page.page_index}-table-1",
                element_type=ElementType.TABLE,
                bbox=BoundingBox(x=0.05, y=0.32, width=0.9, height=0.28),
                reading_order=2,
                metadata={"image_path": page.image_path, "tool": self.name},
            ),
            LayoutRegion(
                region_id=f"p{page.page_index}-figure-1",
                element_type=ElementType.FIGURE,
                bbox=BoundingBox(x=0.1, y=0.66, width=0.8, height=0.24),
                reading_order=3,
                metadata={"image_path": page.image_path, "tool": self.name},
            ),
        ]
        edges = [(regions[index].region_id, regions[index + 1].region_id) for index in range(len(regions) - 1)]
        return PageLayoutGraph(
            session_id=context.session_id,
            page_index=page.page_index,
            regions=regions,
            edges=edges,
        )

    def recognize_text(self, context: OCRToolContext) -> ParsedElement:
        region = context.region or self._default_region(context, ElementType.TEXT)
        text = f"Dummy text block for page {context.page.page_index}."
        return ParsedElement(
            element_id=region.region_id,
            element_type=region.element_type,
            page_index=context.page.page_index,
            reading_order=region.reading_order,
            markdown=text,
            html=f"<p>{escape(text)}</p>",
            text=text,
            bbox=region.bbox,
            confidence=region.confidence,
            metadata={"processor": self.__class__.__name__, "image_path": context.page.image_path, "tool": self.name},
        )

    def detect_table(self, context: OCRToolContext) -> ParsedElement:
        region = context.region or self._default_region(context, ElementType.TABLE)
        table_rows = [["Header A", "Header B"], ["Value A", "Value B"]]
        markdown = "| Header A | Header B |\n| --- | --- |\n| Value A | Value B |"
        html = (
            "<table><tr><th>Header A</th><th>Header B</th></tr>"
            "<tr><td>Value A</td><td>Value B</td></tr></table>"
        )
        return ParsedElement(
            element_id=region.region_id,
            element_type=ElementType.TABLE,
            page_index=context.page.page_index,
            reading_order=region.reading_order,
            markdown=markdown,
            html=html,
            data={"rows": table_rows, "cells": []},
            bbox=region.bbox,
            confidence=region.confidence,
            metadata={"processor": self.__class__.__name__, "image_path": context.page.image_path, "tool": self.name},
        )

    def detect_figure(self, context: OCRToolContext) -> ParsedElement:
        region = context.region or self._default_region(context, ElementType.FIGURE)
        summary = f"Dummy visual summary for {region.element_type} on page {context.page.page_index}."
        return ParsedElement(
            element_id=region.region_id,
            element_type=region.element_type,
            page_index=context.page.page_index,
            reading_order=region.reading_order,
            markdown=f"![{summary}]({context.page.image_path})",
            html=(
                f'<figure><img src="{escape(context.page.image_path)}" alt="{escape(summary)}">'
                f"<figcaption>{escape(summary)}</figcaption></figure>"
            ),
            text=summary,
            data={"summary": summary, "caption": None},
            bbox=region.bbox,
            confidence=region.confidence,
            metadata={"processor": self.__class__.__name__, "image_path": context.page.image_path, "tool": self.name},
        )

    def _default_region(self, context: OCRToolContext, element_type: ElementType) -> LayoutRegion:
        return LayoutRegion(
            region_id=f"p{context.page.page_index}-{element_type.value}-1",
            element_type=element_type,
            bbox=BoundingBox(x=0.0, y=0.0, width=1.0, height=1.0),
            reading_order=1,
            metadata={"image_path": context.page.image_path, "tool": self.name},
        )
