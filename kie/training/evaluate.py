"""Evaluate manifest predictions without a model, keeping blanks and tables visible."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import statistics
import unicodedata

from .common import dump_json, load_json, read_manifest, validate_label

MISSING = object()


def flatten(value, path=()):
    if isinstance(value, dict) and value:
        return {p: v for k, child in value.items() for p, v in flatten(child, path + (k,)).items()}
    if isinstance(value, list) and value:
        return {p: v for i, child in enumerate(value) for p, v in flatten(child, path + (i,)).items()}
    return {path: value}


def equal(expected, predicted):
    if isinstance(expected, str) and isinstance(predicted, str):
        return unicodedata.normalize("NFC", expected) == unicodedata.normalize("NFC", predicted)
    return type(expected) is type(predicted) and expected == predicted


def category(path, value):
    if value is None:
        return "null"
    if type(value) is bool:
        return "checkbox"
    if any(type(p) is int for p in path):
        return "array_cell"
    return "nonempty_scalar"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/tt32/test.jsonl"))
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    rows = read_manifest(args.manifest)
    if not rows:
        parser.error("Manifest is empty")
    documents, buckets, by_form = [], defaultdict(lambda: [0, 0]), defaultdict(list)
    total_correct = total_paths = valid_json = valid_schema = invented_blanks = blank_count = 0
    for row in rows:
        path = args.predictions / f"{row['id']}.json"
        parsed = schema_ok = False
        try:
            prediction = load_json(path)
            parsed = True
            validate_label(prediction, row["template"])
            schema_ok = True
        except (OSError, ValueError):
            pass
        valid_json += parsed
        valid_schema += schema_ok
        expected = flatten(row["label"])
        actual = flatten(prediction) if parsed else {}
        union = set(expected) | set(actual)
        correct = sum(equal(expected.get(p, MISSING), actual.get(p, MISSING)) for p in union)
        accuracy = correct / len(union) if union else 1.0
        for p, value in expected.items():
            bucket = buckets[category(p, value)]
            bucket[0] += equal(value, actual.get(p, MISSING))
            bucket[1] += 1
            if value is None:
                blank_count += 1
                invented_blanks += p in actual and actual[p] is not None
        documents.append({"id": row["id"], "form": row["form"], "json_valid": parsed,
                          "schema_valid": schema_ok, "exact_accuracy": accuracy,
                          "correct": correct, "paths": len(union)})
        by_form[row["form"]].append(accuracy)
        total_correct += correct
        total_paths += len(union)
    metadata_path = args.predictions / "generation_metadata.json"
    latency = None
    if metadata_path.exists():
        metadata = load_json(metadata_path)["documents"]
        times = sorted(m["seconds"] for m in metadata if m["id"] in {r["id"] for r in rows})
        if times:
            latency = {"median_seconds": statistics.median(times),
                       "p95_seconds": times[min(len(times) - 1, int(len(times) * 0.95))]}
    report = {"documents": documents,
              "document_macro_exact": statistics.mean(d["exact_accuracy"] for d in documents),
              "form_macro_exact": statistics.mean(statistics.mean(v) for v in by_form.values()),
              "leaf_micro_exact": total_correct / total_paths if total_paths else 1.0,
              "json_valid_rate": valid_json / len(rows), "schema_valid_rate": valid_schema / len(rows),
              "field_accuracy": {k: {"correct": v[0], "count": v[1], "accuracy": v[0] / v[1]}
                                 for k, v in buckets.items()},
              "blank_hallucination_rate": invented_blanks / blank_count if blank_count else None,
              "latency": latency,
              "metric_notes": "Exact strings normalize Unicode NFC only. Array positions matter. "
                              "Missing/extra paths and invalid/missing predictions score zero. "
                              "Field categories are mutually exclusive; null and checkbox precede array_cell."}
    dump_json(args.report, report)
    print(f"Document macro exact: {report['document_macro_exact']:.4f}; "
          f"JSON valid: {report['json_valid_rate']:.4f}; schema valid: {report['schema_valid_rate']:.4f}")


if __name__ == "__main__":
    main()
