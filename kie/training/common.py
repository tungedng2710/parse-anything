"""Schema conversion and manifest helpers; no ML dependencies."""
from __future__ import annotations

import hashlib
import json
import unicodedata
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def normalized(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def convert_schema(schema: dict) -> tuple[dict, str]:
    """Convert the local fields/type format, preserving descriptions separately."""
    descriptions = []

    def convert(fields: list, parent: tuple = ()) -> dict:
        result = {}
        for field in fields:
            key, kind = field["key"], field["type"]
            if key in result:
                raise ValueError(f"Duplicate schema key at {parent + (key,)}")
            path = parent + (key,)
            if field.get("description"):
                descriptions.append({"path": list(path), "description": field["description"]})
            if kind == "object":
                result[key] = convert(field["fields"], path)
            elif kind == "table":
                result[key] = [convert(field["fields"], path + ("[]",))]
            elif kind == "list":
                result[key] = ["verbatim-string"]
            elif kind in {"string", "boolean", "integer", "number"}:
                result[key] = "verbatim-string" if kind == "string" else kind
            else:
                raise ValueError(f"Unsupported schema type {kind!r} at {path}")
        return result

    template = convert(schema["fields"])
    instructions = (
        "Extract only evidence visible in the supplied document. Follow the field rules below. "
        "Preserve identifiers, leading zeros, punctuation, array order and printed symbols. "
        "Use null for blank scalar fields and [] for absent lists; an unchecked checkbox is "
        "false, not null. Do not complete clipped text or infer missing clinical information. "
        "Rules use path arrays; [] denotes each table row.\n"
        + json.dumps(descriptions, ensure_ascii=False)
    )
    return template, instructions


def validate_label(value: Any, template: Any, path: tuple = ()) -> None:
    """Require exact schema shape, including nulls, arrays and real booleans."""
    if isinstance(template, dict):
        if not isinstance(value, dict) or set(value) != set(template):
            raise ValueError(f"Object keys differ at {path}")
        for key, child in template.items():
            validate_label(value[key], child, path + (key,))
    elif isinstance(template, list):
        if not isinstance(value, list):
            raise ValueError(f"Expected array at {path}")
        for index, child in enumerate(value):
            validate_label(child, template[0], path + (index,))
    elif value is not None:
        valid = {
            "verbatim-string": isinstance(value, str),
            "boolean": type(value) is bool,
            "integer": type(value) is int,
            "number": type(value) in (int, float),
        }
        if not valid.get(template, False):
            raise ValueError(f"Wrong scalar type at {path}: expected {template}")


def read_manifest(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate example IDs in {path}")
    return rows


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def verify_disjoint(*splits: list[dict]) -> None:
    for index, left in enumerate(splits):
        for right in splits[index + 1:]:
            for key in ("id", "group", "pdf_sha256"):
                if {r[key] for r in left} & {r[key] for r in right}:
                    raise ValueError(f"Split leakage detected by {key}")


def user_messages(row: dict, images: list, input_mode: str) -> list[dict]:
    content = [{"type": "image", "image": image} for image in images]
    if input_mode == "image-text":
        content.append({"type": "text", "text": "PDF text layer (may omit visual evidence):\n" + row["text"]})
    return [{"role": "user", "content": content}]


def template_kwargs(row: dict) -> dict:
    return {"template": json.dumps(row["template"], ensure_ascii=False),
            "instructions": row["instructions"], "enable_thinking": False}
