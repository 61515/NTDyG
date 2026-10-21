#!/usr/bin/env python3
"""Exercise preprocessing, memory training, graph construction, and evaluation on synthetic events."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "smoke")
    p.add_argument("--variants", nargs="+", choices=["full", "A1", "A2", "A3"], default=["full"])
    args = p.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(20261021)
    features = json.loads((ROOT / "configs" / "feature_columns.json").read_text())["IIoTAPT"]
    n = 120
    values = rng.normal(size=(n, len(features)))
    labels = (np.arange(n) % 5 == 0).astype(np.int64)
    values[:, :4] += 0.7 * labels[:, None]
    frame = pd.DataFrame(values, columns=features)
    frame["Timestamp_Label"] = np.arange(n)
    frame["Label_encoded"] = labels
    frame["Src IP Label"] = np.arange(n) % 8
    frame["Dst IP Label"] = (np.arange(n) * 3 + 1) % 8
    frame.loc[frame.index >= 100, "Src IP Label"] = 8
    raw = out / "synthetic.csv"
    frame.to_csv(raw, index=False)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "prepare_data.py"),
                    "--input", str(raw), "--dataset", "IIoTAPT", "--order-column", "Timestamp_Label",
                    "--output-dir", str(out / "data"), "--split-dir", str(out / "splits")], check=True)
    results = []
    for variant in args.variants:
        subprocess.run([sys.executable, str(ROOT / "code" / "run_candidate_experiment.py"),
                        "--dataset", "IIoTAPT", "--ablation", variant, "--device", "cpu",
                        "--memory-epochs", "1", "--model-epochs", "1", "--seed", "95545",
                        "--data-dir", str(out / "data"), "--split-dir", str(out / "splits"),
                        "--output-dir", str(out / "runs")], check=True)
        run = out / "runs" / ("candidate_60_20_20_IIoTAPT_" + variant + "_seed_95545")
        result = json.loads((run / "results.json").read_text())
        assert result["test_evaluation_calls"] == 1
        assert result["test_metrics"]["events"] == 24
        subprocess.run([sys.executable, str(ROOT / "code" / "evaluate_scores.py"),
                        "--scores", str(run / "test_scores.csv"),
                        "--threshold", str(result["validation_threshold_selected_by_f1"]),
                        "--output", str(run / "recomputed_metrics.json")], check=True)
        recomputed = json.loads((run / "recomputed_metrics.json").read_text())
        for key in ("roc_auc", "average_precision", "f1"):
            assert abs(recomputed[key] - result["test_metrics"][key]) < 1e-9
        results.append({"variant": variant, "passed": True, "events": 24,
                        "parameter_count": result["classifier_parameter_count"]})
    assert len({x["parameter_count"] for x in results}) == 1
    report = {"passed": True, "data": "synthetic only; not paper results", "runs": results}
    (out / "smoke_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
