#!/usr/bin/env python3
"""Audit and summarize retained-protocol runs, including incomplete cells."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def summarize(root):
    protocol = json.loads((root/"protocol.json").read_text())
    rows, errors, cells = [], [], []
    for dataset in protocol["datasets"]:
        for model in protocol["models"]:
            values = []
            for seed in protocol["seeds"]:
                run = root/dataset/model/("seed_"+str(seed))
                path = run/"results.json"
                if not path.exists():
                    continue
                try:
                    result = json.loads(path.read_text())
                    metadata = json.loads((run/"run_metadata.json").read_text())
                    history = pd.read_csv(run/"epoch_metrics.csv")
                    scores = pd.read_csv(run/"selected_test_scores.csv")
                    measured = dict(roc_auc=float(roc_auc_score(scores.label, scores.score)),
                                    average_precision=float(average_precision_score(scores.label, scores.score)))
                    selected = history.iloc[int(history.roc_auc.to_numpy().argmax())]
                    if int(selected.epoch) != result["selected_epoch"]:
                        raise ValueError("selected epoch is not the earliest maximum test AUC")
                    for key in measured:
                        if not np.isclose(measured[key], result["selected_test_metrics"][key], atol=1e-10, rtol=0):
                            raise ValueError("score-vector audit failed: "+key)
                        if not np.isclose(float(selected[key]), measured[key], atol=1e-10, rtol=0):
                            raise ValueError("selected checkpoint metric audit failed: "+key)
                    if len(scores) != metadata["partition_counts"]["test"]["events"]:
                        raise ValueError("test event count mismatch")
                    for key, path_key in (("source_csv_sha256", "source_csv"),
                                          ("train_index_sha256", "train_index_file"),
                                          ("test_index_sha256", "test_index_file")):
                        h = hashlib.sha256()
                        with Path(metadata[path_key]).open("rb") as stream:
                            for block in iter(lambda: stream.read(8*1024*1024), b""):
                                h.update(block)
                        if h.hexdigest() != metadata[key]:
                            raise ValueError("source/index changed since training: "+path_key)
                    record = dict(dataset=dataset, model=model, seed=seed, **measured,
                                  selected_epoch=result["selected_epoch"], epochs_completed=result["epochs_completed"],
                                  seconds=result["elapsed_seconds"], score_audit_passed=True)
                    rows.append(record)
                    values.append(record)
                except Exception as error:
                    errors.append(dict(dataset=dataset, model=model, seed=seed, error=str(error)))
            cell = dict(dataset=dataset, model=model, completed=len(values), expected=len(protocol["seeds"]))
            if len(values) == len(protocol["seeds"]):
                for metric in ("roc_auc", "average_precision"):
                    array = np.array([v[metric] for v in values])
                    cell[metric] = dict(mean=float(array.mean()), sample_sd=float(array.std(ddof=1)))
            cells.append(cell)
    report = dict(protocol=protocol["protocol"], completed_runs=len(rows), expected_runs=protocol["planned_runs"],
                  complete=len(rows)==protocol["planned_runs"] and not errors, cells=cells, errors=errors,
                  all_completed_scores_audited=not errors, selection="test ROC-AUC; AP at the same checkpoint",
                  historical_table_reproduction_verified=False)
    (root/"summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    if rows:
        pd.DataFrame(rows).to_csv(root/"per_seed_results.csv", index=False)
    text = ["# 现存旧流程对照进度与结果", "", f"已完成并核验 {len(rows)}/{protocol['planned_runs']} 次运行。", "",
            "采用现存 1 倍旧索引、原标准化和训练损失、正常样本训练、打乱目标样本、原学习率/batch、最多 200 轮及测试 AUC 选模/早停。AP 来自相同 checkpoint。没有重跑 NTDyG，也未把这些结果声明为原稿流水线的已核实重建或独立测试估计。", "",
            "| 数据集 | 方法 | 完成种子 | ROC-AUC (%) | AP (%) |", "|---|---|---|---|---|"]
    for cell in cells:
        metric_text = []
        for key in ("roc_auc", "average_precision"):
            stat = cell.get(key)
            if stat:
                digits = 3 if 0 < stat['sample_sd']*100 < .005 else 2
                metric_text.append(f"{100*stat['mean']:.{digits}f} ± {100*stat['sample_sd']:.{digits}f}")
            else:
                metric_text.append("运行中")
        text.append(f"| {cell['dataset']} | {cell['model']} | {cell['completed']}/{cell['expected']} | {metric_text[0]} | {metric_text[1]} |")
    if errors:
        text.extend(["", "核验错误：", "", "```json", json.dumps(errors, ensure_ascii=False, indent=2), "```"])
    (root/"report.md").write_text("\n".join(text)+"\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.output_dir.resolve())
    print(json.dumps({k:result[k] for k in ("completed_runs", "expected_runs", "complete", "errors")}))
