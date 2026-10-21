#!/usr/bin/env python3
"""Screen selected numeric features for extreme values and label association."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def audit_features(
    csv_path: Path,
    label_column: str,
    columns: list[str],
    threshold: float,
) -> dict[str, Any]:
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"{csv_path}: CSV is empty") from exc

        required = set(columns) | {label_column}
        missing = required - set(header)
        if missing:
            raise ValueError(f"{csv_path}: missing columns: {sorted(missing)}")

        positions = {column: header.index(column) for column in columns}
        label_position = header.index(label_column)
        labels: dict[str, int] = {}
        stats = {
            column: {
                "finite": 0,
                "non_numeric_or_nonfinite": 0,
                "abs_at_least_threshold": 0,
                "extreme_by_label": {},
                "min": None,
                "max": None,
                "examples": [],
            }
            for column in columns
        }
        row_count = 0
        for row_count, row in enumerate(reader, start=1):
            if len(row) != len(header):
                raise ValueError(
                    f"{csv_path}: source row {row_count} has {len(row)} fields; "
                    f"expected {len(header)}"
                )
            label = row[label_position].strip()
            labels[label] = labels.get(label, 0) + 1
            for column, position in positions.items():
                item = stats[column]
                try:
                    value = float(row[position])
                except ValueError:
                    item["non_numeric_or_nonfinite"] += 1
                    continue
                if not math.isfinite(value):
                    item["non_numeric_or_nonfinite"] += 1
                    continue

                item["finite"] += 1
                item["min"] = value if item["min"] is None else min(item["min"], value)
                item["max"] = value if item["max"] is None else max(item["max"], value)
                if abs(value) >= threshold:
                    item["abs_at_least_threshold"] += 1
                    by_label = item["extreme_by_label"]
                    by_label[label] = by_label.get(label, 0) + 1
                    if len(item["examples"]) < 3:
                        item["examples"].append(
                            {"source_row_zero_based": row_count - 1, "label": label, "value": value}
                        )

    return {
        "source_csv": str(csv_path),
        "source_rows": row_count,
        "label_column": label_column,
        "label_counts": labels,
        "absolute_value_threshold": threshold,
        "screened_columns": stats,
        "interpretation": (
            "Extreme values are a screening signal only; this report does not establish "
            "feature semantics or prove target leakage."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--label-column", required=True)
    parser.add_argument("--columns", required=True, help="Comma-separated feature names")
    parser.add_argument("--threshold", type=float, default=1e9)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.threshold <= 0:
        raise SystemExit("--threshold must be positive")
    columns = [column.strip() for column in args.columns.split(",") if column.strip()]
    if not columns:
        raise SystemExit("--columns must contain at least one feature name")
    report = audit_features(args.csv, args.label_column, columns, args.threshold)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
