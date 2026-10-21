#!/usr/bin/env python3
"""Independently verify all selected scores before exporting Table VI cells."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from run_legacy_multiclass import CODE, MODELS, CONFIG, SEEDS, RATES, write_json


def independently_recompute(labels, logits):
    # Separate class-by-class implementation, rather than the training metric helper.
    aucs = [roc_auc_score(labels == c, logits[:, c]) for c in range(logits.shape[1])]
    aps = [average_precision_score(labels == c, logits[:, c]) for c in range(logits.shape[1])]
    indicator = np.eye(logits.shape[1], dtype=np.int8)[labels]
    answer = dict(macro_ovr_auc=float(np.mean(aucs)), macro_ovr_ap=float(np.mean(aps)),
                  micro_ovr_auc=float(roc_auc_score(indicator.ravel(), logits.ravel())))
    if logits.shape[1] == 2:
        answer.update(roc_auc=float(roc_auc_score(labels, logits[:, 1])),
                      average_precision=float(average_precision_score(labels, logits[:, 1])))
    else:
        answer.update(roc_auc=answer["macro_ovr_auc"], average_precision=answer["macro_ovr_ap"])
    return answer


def audit(root, require_complete=True):
    root = Path(root)
    protocol = json.loads((root / "protocol.json").read_text())
    assert protocol["seeds"] == list(SEEDS) and protocol["models"] == list(MODELS)
    records, missing, partitions = [], [], {}
    for dataset, rate, model, seed in itertools.product(CONFIG, RATES, MODELS, SEEDS):
        folder = root / dataset / ("rate_" + str(rate)) / model / ("seed_" + str(seed))
        if not (folder / "results.json").exists():
            missing.append(dict(dataset=dataset, rate=rate, model=model, seed=seed))
            continue
        result = json.loads((folder / "results.json").read_text())
        metadata = json.loads((folder / "run_metadata.json").read_text())
        assert result["status"] == "complete" and result["seed"] == seed
        assert result["dataset"] == dataset and result["model"] == model and result["rate_percent"] == rate
        assert 16 <= result["epochs_completed"] <= protocol["max_epochs"], "Preflight must not enter the final table"
        key = dataset, rate, seed
        if key not in partitions:
            split = root / "splits" / dataset / ("rate_" + str(rate)) / ("seed_" + str(seed))
            with np.load(split / "prepared_arrays.npz") as arrays:
                n = metadata["split"]["train_count"]
                partitions[key] = (arrays["labels"][n:], arrays["source_rows"][n:])
        labels, source_rows = partitions[key]
        with np.load(folder / "selected_test_scores.npz") as scores:
            np.testing.assert_array_equal(scores["labels"], labels)
            np.testing.assert_array_equal(scores["source_rows"], source_rows)
            logits = scores["logits"]
        assert logits.shape == (len(labels), metadata["split"]["num_classes"])
        assert np.isfinite(logits).all()
        recomputed = independently_recompute(labels, logits)
        for metric, value in recomputed.items():
            assert abs(value - result["selected_test_metrics"][metric]) < 1e-12, (folder, metric)
        epochs = pd.read_csv(folder / "epoch_metrics.csv")
        selected = epochs.iloc[int(np.argmax(epochs.roc_auc.to_numpy()))]
        assert int(selected.epoch) == result["selected_epoch"]
        assert len(epochs) == result["epochs_completed"]
        for metric, value in recomputed.items():
            assert abs(value - float(selected[metric])) < 1e-12, (folder, metric, "different checkpoint")
        records.append(dict(dataset=dataset, rate_percent=rate, model=model, seed=seed,
                            selected_epoch=result["selected_epoch"], epochs_completed=result["epochs_completed"],
                            **recomputed, elapsed_seconds=result["elapsed_seconds"],
                            train_test_overlap=metadata["split"]["train_test_overlap"]))
    if records:
        pd.DataFrame(records).to_csv(root / "audited_runs.csv", index=False)
    report = dict(expected_runs=96, verified_runs=len(records), missing_runs=missing,
                  status="complete" if not missing else "incomplete", failed_score_checks=0,
                  original_ntdyg_rerun=False,
                  historical_partition_reproduction_verified=False,
                  multiclass_classes=dict(CloudDDoS=26, Darknet=8), IIoTAPT_retained_classes=2,
                  run_scores_independently_recomputed=True)
    write_json(root / "audit.json", report)
    if missing and require_complete:
        raise RuntimeError(f"Only {len(records)}/96 verified runs; final table export refused")
    groups = []
    for model, dataset, rate in itertools.product(MODELS, CONFIG, (5, 1)):
        rows = [r for r in records if r["model"] == model and r["dataset"] == dataset and r["rate_percent"] == rate]
        if len(rows) != 4:
            continue
        assert {r["seed"] for r in rows} == set(SEEDS)
        group = dict(model=model, dataset=dataset, rate_percent=rate)
        for metric in ("roc_auc", "average_precision"):
            values = np.array([r[metric] for r in sorted(rows, key=lambda row: row["seed"])]) * 100
            group[metric + "_mean"] = float(values.mean())
            group[metric + "_sample_sd"] = float(values.std(ddof=1))
        groups.append(group)
    if groups:
        pd.DataFrame(groups).to_csv(root / "summary.csv", index=False)
    if missing:
        return report, groups
    assert len(groups) == 24
    latex_rows = []
    for model in MODELS:
        cells = []
        for dataset, rate in itertools.product(CONFIG, (5, 1)):
            group = next(g for g in groups if (g["model"], g["dataset"], g["rate_percent"]) == (model, dataset, rate))
            for metric in ("roc_auc", "average_precision"):
                cells.append(r"\rev{$%.2f\pm%.2f$\%%}" % (group[metric + "_mean"], group[metric + "_sample_sd"]))
        latex_rows.append(r"\rev{" + model + "} & " + " & ".join(cells) + r" \\")
    (root / "table_vi_rows.tex").write_text("\n".join(latex_rows) + "\n")
    lines = ["# Table VI: verified limited-label additions", "",
             "All 96 registered runs completed. Numbers are percentages, mean ± sample SD across seeds 95545–95548.", "",
             "| Method | Dataset | Labeled ratio | AUC (%) | AP / mAP (%) |",
             "|---|---|---:|---:|---:|"]
    for group in groups:
        lines.append("| {model} | {dataset} | {rate_percent}% | {roc_auc_mean:.2f} ± {roc_auc_sample_sd:.2f} | {average_precision_mean:.2f} ± {average_precision_sample_sd:.2f} |".format(**group))
    lines += ["", "CloudDDoS uses 26 retained labels; Darknet uses eight application-category labels. IIoTAPT has only two retained labels, so its AUC/AP remain positive-class binary metrics.", "",
              "The saved 1%/5% whole-source draws restore the rule declared in the original manuscript; they are not recovered historical draws. Retained test indices and test-AUC selection are preserved, with overlap counts in the protocol. These scores are not independent test estimates. Original 14-method values, including NTDyG, were not rerun or replaced.", ""]
    (root / "report.md").write_text("\n".join(lines))
    return report, groups


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    report, _ = audit(args.output_dir, require_complete=not args.allow_incomplete)
    print(json.dumps({k: v for k, v in report.items() if k != "missing_runs"}, ensure_ascii=False, indent=2))
