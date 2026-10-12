"""Audit local PDFs and prepare conservative, grouped train/validation/test splits."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import math
from pathlib import Path
import random
import re

from .common import (convert_schema, digest, dump_json, load_json, normalized,
                     validate_label, verify_disjoint, write_manifest)


def patient_tokens(label: dict) -> set[str]:
    """Conservative linkage from patient IDs and names inside patient blocks.

    This is heuristic, not a replacement for audited patient/encounter metadata.
    Names deliberately over-group homonyms rather than expose them across splits.
    """
    tokens = set()

    def walk(value, parents=()):
        if isinstance(value, dict):
            for key, child in value.items():
                key_norm = normalized(key).casefold()
                context = " ".join(parents).casefold()
                patient_id = any(term in key_norm for term in (
                    "mã bệnh nhân", "mã người bệnh", "mã số người bệnh", "mã bn"))
                patient_name = ("bệnh nhân" in context or "người bệnh" in context) and key_norm in {
                    "họ tên", "họ và tên", "tên bệnh nhân", "họ tên người bệnh"}
                if isinstance(child, str) and child.strip() and (patient_id or patient_name):
                    compact = re.sub(r"\s+", "", normalized(child).casefold())
                    tokens.add(digest((("id:" if patient_id else "name:") + compact).encode()))
                walk(child, parents + (key,))
        elif isinstance(value, list):
            for child in value:
                walk(child, parents)

    walk(label)
    return tokens


def assign_groups(rows: list[dict], extra_groups: dict) -> None:
    parents = list(range(len(rows)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    seen = {}
    for i, row in enumerate(rows):
        keys = ["form:" + row["form"], "pdf:" + row["pdf_sha256"]]
        keys.extend("patient:" + token for token in row.pop("patient_tokens"))
        if row["source_pdf"] in extra_groups:
            keys.append("external:" + str(extra_groups[row["source_pdf"]]))
        for key in keys:
            if key in seen:
                parents[find(i)] = find(seen[key])
            else:
                seen[key] = i
    members = defaultdict(list)
    for i, row in enumerate(rows):
        members[find(i)].append(row["id"])
    for i, row in enumerate(rows):
        row["group"] = digest("|".join(sorted(members[find(i)])).encode())[:16]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/v2_TT32_forms"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/tt32"))
    parser.add_argument("--dpi", type=int, default=170)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--exclude-forms", nargs="*", default=["5", "37"],
                        help="Unresolved labels excluded by default; empty flag includes all")
    parser.add_argument("--group-map", type=Path,
                        help="JSON mapping NFC relative PDF paths to audited patient/encounter groups")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.dpi <= 0 or not (0 < args.validation_fraction < 1 and 0 < args.test_fraction < 1
                            and args.validation_fraction + args.test_fraction < 1):
        parser.error("Require positive DPI and positive split fractions summing to less than 1")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        parser.error("Output is nonempty; use a new directory or --overwrite")
    import fitz

    root = args.data_root.resolve()
    index = {}
    for path in root.rglob("*"):
        if path.is_file():
            key = normalized(path.relative_to(root).as_posix())
            if key in index:
                raise ValueError(f"Unicode-normalized path collision: {key}")
            index[key] = path
    mapping = load_json(root / "kie_schema_map.json")
    extra = {normalized(k): v for k, v in load_json(args.group_map).items()} if args.group_map else {}
    unknown = set(extra) - {normalized(k) for k in mapping}
    if unknown:
        raise ValueError(f"Group map contains {len(unknown)} unmapped PDF paths")
    rows, excluded, counts, page_counts = [], [], Counter(), Counter()
    schema_hashes = {}

    def count_leaves(value):
        if isinstance(value, dict):
            for child in value.values():
                count_leaves(child)
        elif isinstance(value, list):
            if not value:
                counts["empty_arrays"] += 1
            for child in value:
                count_leaves(child)
        else:
            counts[type(value).__name__] += 1

    for pdf_key, schema_key in sorted(mapping.items()):
        pdf = index[normalized(pdf_key)]
        schema = index[normalized(schema_key)]
        stem = schema.name.removesuffix(".kie_schema.json")
        candidates = [schema.with_name(stem + suffix) for suffix in (".data.json", ".json")]
        labels = [path for path in candidates if path.is_file()]
        if len(labels) != 1:
            raise ValueError(f"Require one unambiguous label for {schema}: found {len(labels)}")
        label = load_json(labels[0])
        template, instructions = convert_schema(load_json(schema))
        validate_label(label, template)
        count_leaves(label)
        form = pdf.parent.name.split("_")[0]
        doc_id = f"{form}-{digest(normalized(pdf_key).encode())[:12]}"
        with fitz.open(pdf) as document:
            if document.needs_pass or not len(document):
                raise ValueError(f"PDF is encrypted or empty: {pdf}")
            pages = len(document)
            page_counts[pages] += 1
            text = "\n\n".join(f"--- Page {i + 1} ---\n{p.get_text()}" for i, p in enumerate(document))
            counts["pages"] += pages
            counts["pages_with_text"] += sum(bool(p.get_text().strip()) for p in document)
            if form in args.exclude_forms:
                excluded.append({"id": doc_id, "form": form, "reason": "unresolved prelabel/chart policy"})
                continue
            image_paths = []
            for i, page in enumerate(document):
                target = output / "images" / doc_id / f"{i + 1:03d}.png"
                target.parent.mkdir(parents=True, exist_ok=True)
                page.get_pixmap(dpi=args.dpi, alpha=False).save(target)
                image_paths.append(str(target))
        schema_hashes[normalized(schema_key)] = digest(schema.read_bytes())
        rows.append({"id": doc_id, "form": form, "source_pdf": normalized(pdf_key),
                     "pdf_sha256": digest(pdf.read_bytes()), "schema_sha256": digest(schema.read_bytes()),
                     "label_sha256": digest(labels[0].read_bytes()), "images": image_paths,
                     "template": template, "instructions": instructions, "label": label, "text": text,
                     "label_status": "reviewed_prelabel_per_local_report",
                     "patient_tokens": sorted(patient_tokens(label))})

    assign_groups(rows, extra)
    groups = sorted({row["group"] for row in rows})
    random.Random(args.seed).shuffle(groups)
    n_val = max(1, math.floor(len(groups) * args.validation_fraction))
    n_test = max(1, math.floor(len(groups) * args.test_fraction))
    if len(groups) <= n_val + n_test:
        raise ValueError("Too few independent groups for three splits; collect more data")
    test_groups, val_groups = set(groups[:n_test]), set(groups[n_test:n_test + n_val])
    splits = {"train": [], "validation": [], "test": []}
    for row in rows:
        split = "test" if row["group"] in test_groups else "validation" if row["group"] in val_groups else "train"
        splits[split].append(row)
    verify_disjoint(*splits.values())
    for name, split_rows in splits.items():
        write_manifest(output / f"{name}.jsonl", split_rows)
    report = {"pdfs_on_disk": sum(p.suffix.lower() == ".pdf" for p in index.values()),
              "mapped_documents": len(mapping), "mapped_form_folders": len({normalized(p).split('/')[0] for p in mapping}),
              "unmapped_pdfs": sorted(k for k, p in index.items() if p.suffix.lower() == ".pdf"
                                      and k not in {normalized(x) for x in mapping}),
              "leaf_counts_all_mapped": dict(counts), "page_distribution_all_mapped": dict(page_counts),
              "excluded": excluded, "eligible_documents": len(rows), "independent_groups": len(groups),
              "split_documents": {k: len(v) for k, v in splits.items()},
              "split_groups": {k: len({r['group'] for r in v}) for k, v in splits.items()},
              "seed": args.seed, "dpi": args.dpi, "data_root": str(root),
              "schema_hashes": schema_hashes,
              "limitations": ["Patient linkage is heuristic; audit group-map before reporting scientific results.",
                              "Family-disjoint splits estimate transfer to unseen forms, not within-form generalization.",
                              "GLM prelabels are not independently established gold labels."]}
    dump_json(output / "audit.json", report)
    print(f"Prepared {len(rows)} documents in {len(groups)} groups: {report['split_documents']}")
    print(f"Audit: {output / 'audit.json'}")


if __name__ == "__main__":
    main()
