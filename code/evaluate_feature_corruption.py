#!/usr/bin/env python3
"""Run a fixed-checkpoint feature-channel corruption diagnostic."""

import argparse
import hashlib
import json
import math
import pickle
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

CODE_DIR = Path(__file__).resolve().parent
CLI_ARGS = sys.argv[1:]
sys.path.insert(0, str(CODE_DIR / "ntdyg"))
sys.argv = [sys.argv[0]]

from memory import NodeMemoryModel
from model.CensNet import CensNet
from model.Combine import CombinedModel
from model.Transformer import TransformerTargetEdgeClassifier
from data_pre import DygDataset, UnifiedPadSpec
from run_candidate_experiment import (
    DATASET_CONFIG,
    dynamic_collate,
    load_ordered_splits,
    make_graph_data,
    metric_record,
    replay_split,
    run_model,
    SPLIT_ROOT,
    sha256,
)


def build_classifier(dataset, device):
    config = DATASET_CONFIG[dataset]
    model_config = SimpleNamespace(
        num_classes=2,
        input_dim=config["feature_dim"],
        n_heads=config["heads"],
        drop_out=0.2,
        n_layer=config["layers"],
    )
    return CombinedModel(
        model_config,
        CensNet(17, config["feature_dim"], model_config.drop_out),
        TransformerTargetEdgeClassifier(model_config, device, hidden_size=config["hidden"]),
        node_memory_dim=16,
    ).to(device)


def build_memory_model(dataset, node_count, device, direction_agnostic=False):
    config = DATASET_CONFIG[dataset]
    return NodeMemoryModel(
        num_nodes=node_count,
        feature_dim=config["feature_dim"],
        memory_dim=16,
        message_dim=config["feature_dim"],
        hidden_dim=32,
        classifier_hidden_dim=32,
        device=str(device),
        direction_agnostic_messages=direction_agnostic,
    )


def save_graph(path, graph):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump(graph, stream, protocol=pickle.HIGHEST_PROTOCOL)


def evaluate_graph(model, graph_path, feature_dim, threshold, device, score_path=None):
    dataset = DygDataset(str(graph_path), UnifiedPadSpec(1, 1, 17, feature_dim))
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        collate_fn=dynamic_collate,
    )
    model.eval()
    scores, labels = [], []
    with torch.inference_mode():
        for batch in loader:
            logits = run_model(model, batch, device)
            scores.append(logits[:, 1].cpu().numpy())
            labels.append(batch["labels"].numpy().astype(np.int64))
    labels, scores = np.concatenate(labels), np.concatenate(scores)
    if score_path is not None:
        np.savez_compressed(score_path, labels=labels, scores=scores, threshold=threshold)
    return metric_record(labels, scores, threshold)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASET_CONFIG), default="IIoTAPT")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mask-seed", type=int, default=20261013)
    parser.add_argument("--include-clean", action="store_true",
                        help="Recompute a clean control and verify it against the source run before masking.")
    args = parser.parse_args(CLI_ARGS)

    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        parser.error("refusing to overwrite existing output directory: {}".format(output_dir))
    output_dir.mkdir(parents=True)
    source_run = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    base_results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    source_csv = Path(source_run["source_csv"])
    import run_candidate_experiment as runner
    runner.SPLIT_ROOT = Path(source_run.get("split_dir", str(SPLIT_ROOT / args.dataset))).parent
    if source_run["dataset"] != args.dataset or base_results["ablation"] != "full":
        parser.error("run directory must contain the requested dataset's full-model candidate run")

    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    torch.set_num_threads(1)
    _, _, splits, feature_cols, node2id = load_ordered_splits(args.dataset, source_csv)
    if sha256(source_csv) != source_run["source_sha256"]:
        parser.error("source CSV hash differs from the selected model run")
    if feature_cols != source_run["feature_columns"]:
        parser.error("feature columns differ from the selected model run")

    memory_model = build_memory_model(args.dataset, len(node2id), device)
    memory_checkpoint = run_dir / "best_validation_memory_checkpoint.pt"
    memory_model.load_state_dict(torch.load(memory_checkpoint, map_location="cpu", weights_only=True))
    memory_model.eval()

    terminal_state = torch.load(
        run_dir / "train_terminal_memory.pt", map_location="cpu", weights_only=True
    )
    validation_events, validation_state = replay_split(
        SimpleNamespace(device=str(device)),
        memory_model,
        splits["validation"],
        feature_cols,
        (terminal_state["memory"].to(device), terminal_state["last_update"].to(device)),
    )
    train_events = pd.read_pickle(run_dir / "train_event_features.pkl")
    history = pd.concat([train_events, validation_events], ignore_index=True)
    max_node_id = int(max(history["src_id"].max(), history["dst_id"].max()))
    graph_config = SimpleNamespace(k_hop=1, n_degree=10, max_node_id=max(max_node_id, 1))

    classifier = build_classifier(args.dataset, device)
    classifier_checkpoint = run_dir / "best_validation_checkpoint.pt"
    classifier.load_state_dict(torch.load(classifier_checkpoint, map_location="cpu", weights_only=True))
    threshold = float(base_results["validation_threshold_selected_by_f1"])

    rng = np.random.default_rng(args.mask_seed)
    feature_order = rng.permutation(len(feature_cols))
    conditions = []
    fractions = (0.0, 0.10, 0.25, 0.50) if args.include_clean else (0.10, 0.25, 0.50)
    for fraction in fractions:
        count = int(math.ceil(fraction * len(feature_cols)))
        masked_columns = [feature_cols[index] for index in feature_order[:count]]
        corrupted = splits["test"].copy()
        corrupted.loc[:, masked_columns] = 0.0

        start = time.perf_counter()
        test_events, _ = replay_split(
            SimpleNamespace(device=str(device)),
            memory_model,
            corrupted,
            feature_cols,
            validation_state,
        )
        graph = make_graph_data(graph_config, test_events, history, feature_cols)
        graph_path = output_dir / "graphs" / "test_zero_{:02d}pct.pkl".format(round(fraction * 100))
        save_graph(graph_path, graph)
        metrics = evaluate_graph(
            classifier,
            graph_path,
            DATASET_CONFIG[args.dataset]["feature_dim"],
            threshold,
            device,
            output_dir / "scores_zero_{:02d}pct.npz".format(round(fraction * 100)),
        )
        if fraction == 0:
            for metric in ("roc_auc", "average_precision", "precision", "recall", "f1"):
                if abs(metrics[metric] - base_results["test_metrics"][metric]) > 1e-5:
                    (output_dir / "clean_control_mismatch.json").write_text(json.dumps({
                        "metric": metric, "recomputed": metrics,
                        "expected": base_results["test_metrics"],
                    }, indent=2), encoding="utf-8")
                    raise ValueError("Clean control differs from the source checkpoint: " + metric)
        conditions.append({
            "feature_fraction_zeroed": fraction,
            "columns_zeroed": masked_columns,
            "fill_value": 0.0,
            "fill_semantics": "training-standardized mean",
            "test_metrics_at_clean_validation_threshold": metrics,
            "replay_graph_and_inference_seconds": time.perf_counter() - start,
            "graph_cache": str(graph_path.relative_to(output_dir)),
            "scores": "scores_zero_{:02d}pct.npz".format(round(fraction * 100)),
        })
        print(json.dumps(conditions[-1], ensure_ascii=False), flush=True)

    result = {
        "dataset": args.dataset,
        "seed": base_results["seed"],
        "device": str(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "source_sha256": source_run["source_sha256"],
        "split_index_sha256": {
            name: sha256(runner.SPLIT_ROOT / args.dataset / (name + "_indices.csv"))
            for name in ("train", "validation", "test")
        },
        "evaluator_sha256": sha256(Path(__file__)),
        "split_summary": source_run["split_audit"],
        "feature_columns": feature_cols,
        "mask_seed": args.mask_seed,
        "mask_rule": "Nested random feature-channel masks selected without labels; zeroed only in test events after train-fitted standardization.",
        "protocol": "Memory and classifier checkpoints are frozen. Validation events are replayed cleanly from the saved train terminal state. Each perturbed test stream is replayed sequentially so memory updates consume the corrupted event features. The classifier threshold is fixed from clean validation; no perturbed test labels are used for selection.",
        "clean_test_metrics": base_results["test_metrics"],
        "test_evaluations_per_condition": 1,
        "clean_control_recomputed": args.include_clean,
        "conditions": conditions,
        "memory_checkpoint_sha256": sha256(memory_checkpoint),
        "classifier_checkpoint_sha256": sha256(classifier_checkpoint),
        "run_candidate_protocol": "validation-selected diagnostic; see docs/REPRODUCIBILITY.md",
    }
    (output_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("RESULT:", json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
