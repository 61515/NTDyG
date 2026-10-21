#!/usr/bin/env python3
"""Convert an explicitly specified flow CSV to the NTDyG binary event schema."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESERVED = ["Timestamp_Label", "Label_encoded", "Src IP Label", "Dst IP Label"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--dataset", choices=["CloudDDoS", "IIoTAPT", "Darknet"], required=True)
    p.add_argument("--source-column", default="Src IP Label")
    p.add_argument("--destination-column", default="Dst IP Label")
    p.add_argument("--label-column", default="Label_encoded")
    order = p.add_mutually_exclusive_group(required=True)
    order.add_argument("--order-column", help="Numeric event-order key in the source CSV")
    order.add_argument("--use-row-order", action="store_true", help="Explicitly use source row order")
    p.add_argument("--normal-label", default="0")
    p.add_argument("--positive-label", default="1")
    p.add_argument("--feature-columns", type=Path,
                   help="JSON list, or JSON object indexed by dataset; default: configs/feature_columns.json")
    p.add_argument("--output-dir", type=Path, default=ROOT / "data" / "binary")
    p.add_argument("--split-dir", type=Path, default=ROOT / "splits" / "custom")
    p.add_argument("--train-ratio", type=float, default=0.6)
    p.add_argument("--validation-ratio", type=float, default=0.2)
    args = p.parse_args()
    feature_file = args.feature_columns or ROOT / "configs" / "feature_columns.json"
    features = json.loads(feature_file.read_text())
    if isinstance(features, dict):
        features = features[args.dataset]
    if not isinstance(features, list) or len(features) != len(set(features)):
        p.error("Feature columns must be a JSON list of unique column names")
    widths = json.loads((ROOT / "configs" / "model_config.json").read_text())
    if len(features) != widths[args.dataset]["feature_dim"]:
        p.error("Feature count differs from the selected model configuration")
    if set(features) & (set(RESERVED) | {args.source_column, args.destination_column,
                                       args.label_column, args.order_column}):
        p.error("Labels, endpoints, and ordering columns cannot be flow features")
    output = args.output_dir / (args.dataset + "_re.csv")
    split = args.split_dir / args.dataset
    if output.exists() or split.exists():
        p.error("Use new output and split directories; existing files are never overwritten")
    frame = pd.read_csv(args.input)
    required = set(features + [args.source_column, args.destination_column, args.label_column])
    if args.order_column:
        required.add(args.order_column)
    missing = required - set(frame.columns)
    if missing:
        p.error("Missing columns: " + ", ".join(sorted(missing)))
    if frame.empty:
        p.error("Input CSV is empty")
    if frame[[args.source_column, args.destination_column, args.label_column]].isna().any().any():
        p.error("Endpoints and labels must not be missing")
    normal, positive = str(args.normal_label), str(args.positive_label)
    if normal == positive:
        p.error("Normal and positive labels must differ")
    # Numeric labels such as 0 and 0.0 are treated equivalently.
    def label_key(value):
        try:
            return str(float(value))
        except (TypeError, ValueError):
            return str(value)
    mapping = {label_key(normal): 0, label_key(positive): 1}
    labels = frame[args.label_column].map(label_key).map(mapping)
    if labels.isna().any():
        p.error("Input contains unmapped classes; specify an explicit binary mapping before conversion")
    numeric = frame[features].apply(pd.to_numeric, errors="raise").astype(np.float64)
    if not np.isfinite(numeric.to_numpy()).all():
        p.error("Flow features contain NaN/Inf; resolve the source fields explicitly")
    keys = (pd.Series(np.arange(len(frame)), index=frame.index) if args.use_row_order
            else pd.to_numeric(frame[args.order_column], errors="raise"))
    if not np.isfinite(keys.to_numpy(dtype=np.float64)).all():
        p.error("Event-order keys must be finite")
    order_rows = np.argsort(keys.to_numpy(), kind="stable")
    # Register endpoints in encounter order; this is an indexing map, not a label feature.
    registry = {}
    src_ids, dst_ids = np.empty(len(frame), dtype=np.int64), np.empty(len(frame), dtype=np.int64)
    src_values = frame[args.source_column].astype(str).to_numpy()
    dst_values = frame[args.destination_column].astype(str).to_numpy()
    for i in order_rows:
        for value in (src_values[i], dst_values[i]):
            if value not in registry:
                registry[value] = len(registry)
        src_ids[i], dst_ids[i] = registry[src_values[i]], registry[dst_values[i]]
    converted = numeric.copy()
    converted["Timestamp_Label"] = keys
    converted["Label_encoded"] = labels.astype(np.int64)
    converted["Src IP Label"] = src_ids
    converted["Dst IP Label"] = dst_ids
    converted = converted.iloc[order_rows].reset_index(drop=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    converted.to_csv(output, index=False)
    split.mkdir(parents=True)
    cmd = [sys.executable, str(ROOT / "code" / "make_chronological_splits.py"),
           "--csv", str(output), "--output-dir", str(split),
           "--order-column", "Timestamp_Label", "--label-column", "Label_encoded",
           "--train-ratio", str(args.train_ratio), "--validation-ratio", str(args.validation_ratio)]
    subprocess.run(cmd, check=True)
    subprocess.run([sys.executable, str(ROOT / "code" / "audit_splits.py"),
                    "--csv", str(output), "--train-index", str(split / "train_indices.csv"),
                    "--validation-index", str(split / "validation_indices.csv"),
                    "--test-index", str(split / "test_indices.csv"),
                    "--order-column", "Timestamp_Label", "--label-column", "Label_encoded",
                    "--output", str(split / "split_audit.json")], check=True)
    def digest(path):
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024**2), b""):
                h.update(block)
        return h.hexdigest()
    record = {"input_sha256": digest(args.input), "output_sha256": digest(output),
              "dataset": args.dataset, "events": len(converted), "entities": len(registry),
              "feature_columns": features, "feature_scaling": "deferred to training; fit on train only",
              "binary_mapping": {normal: 0, positive: 1},
              "order_source": "source row order" if args.use_row_order else args.order_column,
              "order_semantics": "event-order key; no physical-time meaning inferred",
              "index_semantics": "zero-based rows of the converted, ordered CSV"}
    output.with_suffix(".preprocessing.json").write_text(json.dumps(record, indent=2) + "\n")
    print("Prepared:", output)
    print("Train with --data-dir", args.output_dir, "--split-dir", args.split_dir)


if __name__ == "__main__":
    main()
