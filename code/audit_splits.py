#!/usr/bin/env python3
"""Audit row-index splits and event-order ranges without changing the data."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from decimal import Decimal, InvalidOperation
from itertools import combinations
from pathlib import Path
from typing import Any


def read_indices(path: Path) -> list[int]:
    indices = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.reader(handle):
            if not row or not row[0].strip():
                raise ValueError(f"{path}: index file contains an empty value")
            try:
                index = int(row[0].strip())
            except ValueError as exc:
                raise ValueError(f"{path}: invalid integer index {row[0]!r}") from exc
            indices.append(index)
    if not indices:
        raise ValueError(f"{path}: index file is empty")
    return indices


def _label_key(value: str) -> str:
    try:
        return str(Decimal(value).normalize())
    except InvalidOperation:
        return value.strip()


def _order_key(value: str) -> tuple[str, Any]:
    try:
        return "number", Decimal(value)
    except InvalidOperation:
        try:
            normalized = value.strip().replace("Z", "+00:00")
            return "datetime", datetime.fromisoformat(normalized)
        except ValueError:
            return "text", value


def _order_key_display(value: str) -> str:
    return value


def audit_splits(
    csv_path: Path,
    split_paths: dict[str, Path],
    order_column: str,
    label_column: str,
    positive_label: str = "1",
) -> dict[str, Any]:
    indices = {name: read_indices(path) for name, path in split_paths.items()}
    index_sets = {}
    for name, values in indices.items():
        if len(values) != len(set(values)):
            raise ValueError(f"{name}: duplicate row indices found")
        if min(values) < 0:
            raise ValueError(f"{name}: row indices must be non-negative")
        index_sets[name] = set(values)

    selected_indices = set().union(*index_sets.values())
    selected_rows: dict[int, tuple[str, str]] = {}
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = {order_column, label_column} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{csv_path}: missing required columns: {sorted(missing)}")
        row_count = 0
        for row_count, row in enumerate(reader, start=1):
            index = row_count - 1
            if index in selected_indices:
                selected_rows[index] = (row[order_column], row[label_column])

    out_of_range = sorted(index for index in selected_indices if index >= row_count)
    if out_of_range:
        raise ValueError(
            f"index out of range for {row_count} source rows; "
            f"first invalid index: {out_of_range[0]}"
        )

    overlaps = {
        f"{left}__{right}": len(index_sets[left] & index_sets[right])
        for left, right in combinations(index_sets, 2)
    }
    positive_key = _label_key(positive_label)
    summaries = {}
    for name, values in indices.items():
        rows = [selected_rows[index] for index in values]
        labels = [label for _, label in rows]
        label_counts: dict[str, int] = {}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        anomalous = sum(_label_key(label) == positive_key for label in labels)
        min_key = min((key for key, _ in rows), key=_order_key)
        max_key = max((key for key, _ in rows), key=_order_key)
        summaries[name] = {
            "events": len(rows),
            "normal": len(rows) - anomalous,
            "anomalous": anomalous,
            "anomaly_prevalence": anomalous / len(rows),
            "order_key_min": _order_key_display(min_key),
            "order_key_max": _order_key_display(max_key),
            "label_counts": label_counts,
        }

    ordered = None
    ordered_names = [name for name in ("train", "validation", "test") if name in summaries]
    if len(ordered_names) >= 2:
        ordered = all(
            _order_key(summaries[left]["order_key_max"])
            < _order_key(summaries[right]["order_key_min"])
            for left, right in zip(ordered_names, ordered_names[1:])
        )

    return {
        "source_csv": str(csv_path),
        "source_rows": row_count,
        "indexed_source_rows": len(selected_indices),
        "unindexed_source_rows": row_count - len(selected_indices),
        "order_column": order_column,
        "label_column": label_column,
        "positive_label": positive_label,
        "splits": summaries,
        "pairwise_overlap_counts": overlaps,
        "strict_order_key_order": ordered,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path, help="Source event CSV")
    parser.add_argument("--train-index", required=True, type=Path)
    parser.add_argument("--validation-index", type=Path)
    parser.add_argument("--test-index", required=True, type=Path)
    parser.add_argument("--order-column", "--time-column", dest="order_column", required=True)
    parser.add_argument("--label-column", required=True)
    parser.add_argument("--positive-label", default="1")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    split_paths = {"train": args.train_index}
    if args.validation_index is not None:
        split_paths["validation"] = args.validation_index
    split_paths["test"] = args.test_index

    report = audit_splits(
        csv_path=args.csv,
        split_paths=split_paths,
        order_column=args.order_column,
        label_column=args.label_column,
        positive_label=args.positive_label,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")

    has_overlap = any(report["pairwise_overlap_counts"].values())
    return 1 if has_overlap or report["strict_order_key_order"] is False else 0


if __name__ == "__main__":
    raise SystemExit(main())
