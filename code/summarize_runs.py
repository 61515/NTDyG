#!/usr/bin/env python3
"""Summarize every supplied completed validation-selected run using sample SD."""
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error("Output exists; choose a new filename")
    groups, seen = defaultdict(list), set()
    for path in sorted(a.runs_root.rglob("results.json")):
        record = json.loads(path.read_text())
        if "test_metrics" not in record or "ablation" not in record:
            continue
        key = (record["dataset"], record["ablation"], record["seed"])
        if key in seen:
            raise ValueError("Duplicate dataset/variant/seed: " + repr(key))
        seen.add(key)
        groups[key[:2]].append(record)
    if not groups:
        p.error("No completed validation-selected results found")
    rows = []
    for (dataset, variant), records in sorted(groups.items()):
        row = {"dataset": dataset, "variant": variant, "runs": len(records),
               "seeds": ",".join(str(x["seed"]) for x in records)}
        for name in ("roc_auc", "average_precision", "precision", "recall", "f1"):
            values = [100 * x["test_metrics"][name] for x in records]
            row[name + "_mean_percent"] = statistics.mean(values)
            row[name + "_sample_sd_percent"] = statistics.stdev(values) if len(values) > 1 else ""
        rows.append(row)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Summarized", len(seen), "runs into", a.output)


if __name__ == "__main__":
    main()
