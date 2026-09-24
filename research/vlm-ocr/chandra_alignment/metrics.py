from __future__ import annotations

import json
from typing import Any

from rapidfuzz.distance import Levenshtein


def extract_json(text: str) -> Any | None:
    """Extract one complete JSON response, rejecting trailing generated content."""
    candidate = text.strip()
    if "</think>" in candidate:
        candidate = candidate.split("</think>", 1)[1].strip()
    if candidate.startswith("```"):
        first_newline = candidate.find("\n")
        if first_newline >= 0:
            candidate = candidate[first_newline + 1 :]
        if candidate.rstrip().endswith("```"):
            candidate = candidate.rstrip()[:-3].rstrip()

    decoder = json.JSONDecoder()
    starts = [index for index, character in enumerate(candidate) if character in "[{"]
    start = starts[0] if starts else 0
    try:
        value, end = decoder.raw_decode(candidate[start:])
    except json.JSONDecodeError:
        return None
    trailing = candidate[start + end :].strip()
    return value if trailing in {"", "```"} else None


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def flatten_json(value: Any, path: str = "$") -> set[tuple[str, str]]:
    """Represent JSON leaves as path/value pairs for micro field F1."""
    if isinstance(value, dict):
        leaves: set[tuple[str, str]] = set()
        for key, child in value.items():
            leaves.update(flatten_json(child, f"{path}.{key}"))
        return leaves
    if isinstance(value, list):
        leaves = set()
        for index, child in enumerate(value):
            leaves.update(flatten_json(child, f"{path}[{index}]"))
        return leaves
    return {(path, canonical_json(value))}


def score_json_prediction(prediction: str, reference: Any) -> dict[str, Any]:
    parsed = extract_json(prediction)
    reference_text = canonical_json(reference)
    comparison_text = (
        canonical_json(parsed) if parsed is not None else prediction.strip()
    )
    reference_fields = flatten_json(reference)
    predicted_fields = flatten_json(parsed) if parsed is not None else set()
    true_positives = len(reference_fields & predicted_fields)
    return {
        "json_valid": parsed is not None,
        "exact_match": parsed == reference if parsed is not None else False,
        "edit_similarity": Levenshtein.normalized_similarity(
            comparison_text, reference_text
        ),
        "field_true_positives": true_positives,
        "field_predictions": len(predicted_fields),
        "field_references": len(reference_fields),
        "parsed_prediction": parsed,
    }


def aggregate_scores(scores: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(scores)
    if not count:
        return {
            "samples": 0,
            "json_valid_rate": 0.0,
            "exact_match_rate": 0.0,
            "mean_edit_similarity": 0.0,
            "field_precision": 0.0,
            "field_recall": 0.0,
            "field_f1": 0.0,
        }
    true_positives = sum(score["field_true_positives"] for score in scores)
    predictions = sum(score["field_predictions"] for score in scores)
    references = sum(score["field_references"] for score in scores)
    precision = true_positives / predictions if predictions else 0.0
    recall = true_positives / references if references else 0.0
    field_f1 = (
        2 * precision * recall / (precision + recall) if precision + recall else 0.0
    )
    return {
        "samples": count,
        "json_valid_rate": sum(score["json_valid"] for score in scores) / count,
        "exact_match_rate": sum(score["exact_match"] for score in scores) / count,
        "mean_edit_similarity": sum(score["edit_similarity"] for score in scores)
        / count,
        "field_precision": precision,
        "field_recall": recall,
        "field_f1": field_f1,
    }
