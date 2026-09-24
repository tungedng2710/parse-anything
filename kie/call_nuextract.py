#!/usr/bin/env python3
"""Call a NuExtract3 vLLM server through its OpenAI-compatible API."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import Any

from openai import OpenAI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract structured data or document content with NuExtract3."
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("NUEXTRACT_BASE_URL", "https://8895--main--frontier--idp-lab.coder.vts-ai.space/v1"),
        help="OpenAI-compatible API base URL (default: %(default)s)",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("NUEXTRACT_API_KEY", "EMPTY"),
        help="API key, if the vLLM server requires one",
    )
    parser.add_argument("--model", default="numind/NuExtract3")

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--text", help="Text to process")
    input_group.add_argument(
        "--image",
        action="append",
        type=Path,
        help="Image to process; repeat for multiple pages in page order",
    )
    input_group.add_argument(
        "--pdf",
        action="append",
        type=Path,
        help="PDF to process; repeat for multiple PDFs in document order",
    )

    parser.add_argument(
        "--mode",
        choices=("structured", "markdown", "content", "template-generation"),
        default="structured",
        help="NuExtract operation (default: %(default)s)",
    )
    parser.add_argument(
        "--template",
        type=Path,
        help="Path to a JSON extraction template; required in structured mode",
    )
    parser.add_argument(
        "--instructions",
        help="Additional extraction instructions",
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Enable reasoning for difficult documents",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Sampling temperature (default: 0.2, or 0.6 with --thinking)",
    )
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--dpi",
        type=int,
        default=170,
        help="PDF rendering resolution (default: %(default)s)",
    )
    return parser.parse_args()


def image_data_url(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"Image does not exist or is not a file: {path}")

    mime_type, _ = mimetypes.guess_type(path.name)
    if not mime_type or not mime_type.startswith("image/"):
        raise ValueError(f"Unsupported image type: {path}")

    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def pdf_data_urls(path: Path, dpi: int) -> list[str]:
    if not path.is_file():
        raise ValueError(f"PDF does not exist or is not a file: {path}")
    if dpi <= 0:
        raise ValueError("--dpi must be greater than zero")

    try:
        import fitz
    except ImportError as exc:
        raise ValueError("PDF input requires PyMuPDF: pip install pymupdf") from exc

    data_urls: list[str] = []
    try:
        with fitz.open(path) as document:
            for page in document:
                pixmap = page.get_pixmap(dpi=dpi, alpha=False)
                encoded = base64.b64encode(pixmap.tobytes("png")).decode("ascii")
                data_urls.append(f"data:image/png;base64,{encoded}")
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"Could not render PDF {path}: {exc}") from exc

    if not data_urls:
        raise ValueError(f"PDF contains no pages: {path}")
    return data_urls


def build_content(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.text is not None:
        return [{"type": "text", "text": args.text}]

    if args.pdf is not None:
        urls = [url for path in args.pdf for url in pdf_data_urls(path, args.dpi)]
        return [
            {"type": "image_url", "image_url": {"url": url}} for url in urls
        ]

    return [
        {"type": "image_url", "image_url": {"url": image_data_url(path)}}
        for path in args.image
    ]


def build_chat_template_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"enable_thinking": args.thinking}

    if args.mode == "structured":
        if args.template is None:
            raise ValueError("--template is required when --mode=structured")
        try:
            template = json.loads(args.template.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ValueError(f"Template file does not exist: {args.template}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON template {args.template}: {exc}") from exc
        kwargs["template"] = json.dumps(template, ensure_ascii=False)
    else:
        if args.template is not None:
            raise ValueError("--template can only be used with --mode=structured")
        kwargs["mode"] = args.mode

    if args.instructions:
        kwargs["instructions"] = args.instructions

    return kwargs


def main() -> None:
    args = parse_args()

    try:
        content = build_content(args)
        chat_template_kwargs = build_chat_template_kwargs(args)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc

    client = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url,
        timeout=args.timeout,
    )
    response = client.chat.completions.create(
        model=args.model,
        temperature=(0.6 if args.thinking else 0.2)
        if args.temperature is None
        else args.temperature,
        max_tokens=args.max_tokens,
        messages=[{"role": "user", "content": content}],
        extra_body={"chat_template_kwargs": chat_template_kwargs},
    )

    result = response.choices[0].message.content or ""
    if "</think>" in result:
        result = result.split("</think>", maxsplit=1)[1].strip()
    print(result)


if __name__ == "__main__":
    main()
