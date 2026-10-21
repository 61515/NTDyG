#!/usr/bin/env python3
"""Create deterministic splits by a numeric event-order key, not necessarily time."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


def _decimal(value: str, field: str) -> Decimal:
    try:
        result = Decimal(value.strip())
    except InvalidOperation as exc:
        raise ValueError(f"Invalid numeric {field}: {value!r}") from exc
    if not result.is_finite():
        raise ValueError(f"Non-finite {field}: {value!r}")
    return result


def _label_key(value: str) -> str:
    try:
        return str(Decimal(value.strip()).normalize())
    except InvalidOperation:
        return value.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _nearest_cut(cumulative: list[int], target: float, lower: int, upper: int) -> int:
    """Return the number of equal-key groups assigned to the left split."""
    return min(range(lower, upper + 1), key=lambda cut: (abs(cumulative[cut - 1] - target), cut))


def make_ordered_splits(
    csv_path: Path,
    output_dir: Path,
    order_column: str,
    label_column: str,
    train_ratio: float = 0.6,
    validation_ratio: float = 0.2,
    positive_label: str = "1",
    negative_label: str = "0",
) -> dict[str, Any]:
    if train_ratio <= 0 or validation_ratio <= 0 or train_ratio + validation_ratio >= 1:
        raise ValueError("train_ratio and validation_ratio must be positive and sum to less than 1")

    groups: list[dict[str, Any]] = []
    row_count = 0
    previous_key = None
    order_key_equals_source_row = True
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = {order_column, label_column} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{csv_path}: missing required columns: {sorted(missing)}")

        for row_count, row in enumerate(reader, start=1):
            index = row_count - 1
            order_value = row[order_column]
            order_key = _decimal(order_value, order_column)
            if order_key != Decimal(index):
                order_key_equals_source_row = False
            if previous_key is not None and order_key < previous_key:
                raise ValueError(
                    f"Source rows are not ordered by {order_column} at row {index}: "
                    f"{order_value} follows {previous_key}"
                )
            previous_key = order_key
            label = row[label_column].strip()
            if not label:
                raise ValueError(f"Empty label at source row {index}")

            if not groups or groups[-1]["order_key"] != order_key:
                groups.append(
                    {"order_key": order_key, "order_value": order_value, "indices": [], "labels": Counter()}
                )
            groups[-1]["indices"].append(index)
            groups[-1]["labels"][label] += 1

    if row_count == 0:
        raise ValueError(f"{csv_path}: source CSV has no event rows")
    if len(groups) < 3:
        raise ValueError("At least three distinct event-order-key groups are required")

    cumulative = []
    total = 0
    for group in groups:
        total += len(group["indices"])
        cumulative.append(total)

    train_cut = _nearest_cut(cumulative, row_count * train_ratio, 1, len(groups) - 2)
    validation_cut = _nearest_cut(
        cumulative,
        row_count * (train_ratio + validation_ratio),
        train_cut + 1,
        len(groups) - 1,
    )
    split_groups = {
        "train": groups[:train_cut],
        "validation": groups[train_cut:validation_cut],
        "test": groups[validation_cut:],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    split_summaries = {}
    all_indices = []
    for split_name, selected_groups in split_groups.items():
        indices = [index for group in selected_groups for index in group["indices"]]
        all_indices.extend(indices)
        index_path = output_dir / f"{split_name}_indices.csv"
        with index_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerows((index,) for index in indices)

        label_counts = Counter()
        for group in selected_groups:
            label_counts.update(group["labels"])
        positive_key = _label_key(positive_label)
        negative_key = _label_key(negative_label)
        anomalous = sum(count for label, count in label_counts.items() if _label_key(label) == positive_key)
        normal = sum(count for label, count in label_counts.items() if _label_key(label) == negative_key)
        split_summaries[split_name] = {
            "index_file": index_path.name,
            "events": len(indices),
            "normal": normal,
            "anomalous": anomalous,
            "other_label": len(indices) - normal - anomalous,
            "anomaly_prevalence": anomalous / len(indices),
            "order_key_min": selected_groups[0]["order_value"],
            "order_key_max": selected_groups[-1]["order_value"],
            "label_counts": dict(sorted(label_counts.items())),
        }

    if len(all_indices) != row_count or len(set(all_indices)) != row_count:
        raise RuntimeError("Generated splits do not form a disjoint partition of the source rows")

    report = {
        "source_csv": str(csv_path),
        "source_sha256": _sha256(csv_path),
        "source_rows": row_count,
        "order_key_groups": len(groups),
        "order_key_equals_zero_based_source_row": order_key_equals_source_row,
        "order_column": order_column,
        "order_key_semantics": "numeric event-order key; physical-time meaning and units not established",
        "label_column": label_column,
        "positive_label": positive_label,
        "negative_label": negative_label,
        "requested_ratios": {
            "train": train_ratio,
            "validation": validation_ratio,
            "test": 1.0 - train_ratio - validation_ratio,
        },
        "actual_ratios": {
            name: split["events"] / row_count for name, split in split_summaries.items()
        },
        "splits": split_summaries,
        "pairwise_overlap_counts": {"train__validation": 0, "train__test": 0, "validation__test": 0},
        "covers_all_source_rows": True,
        "strict_order_key_order": all(
            _decimal(split_summaries[left]["order_key_max"], order_column)
            < _decimal(split_summaries[right]["order_key_min"], order_column)
            for left, right in (("train", "validation"), ("validation", "test"))
        ),
    }
    report_path = output_dir / "split_summary.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path, help="Source event CSV ordered by the selected key")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--order-column", "--time-column", dest="order_column", required=True,
        help="Numeric event-order key; do not interpret as physical time without source evidence",
    )
    parser.add_argument("--label-column", required=True)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--positive-label", default="1")
    parser.add_argument("--negative-label", default="0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = make_ordered_splits(
        csv_path=args.csv,
        output_dir=args.output_dir,
        order_column=args.order_column,
        label_column=args.label_column,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        positive_label=args.positive_label,
        negative_label=args.negative_label,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
