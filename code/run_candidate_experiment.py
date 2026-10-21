#!/usr/bin/env python3
"""Run an isolated train/validation/test NTDyG candidate experiment.

Validation-selected NTDyG with fresh memory training and ordered classifier
batches. The release is based on the saved CloudDDoS ablation source snapshot.
See docs/REPRODUCIBILITY.md for the scope of the different paper protocols.
"""

import argparse
import hashlib
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from torch import nn
from torch.utils.data import DataLoader


CODE_DIR = Path(__file__).resolve().parent
NTDYG_DIR = CODE_DIR / "ntdyg"
CLI_ARGS = sys.argv[1:]
sys.path.insert(0, str(NTDYG_DIR))
sys.argv = [sys.argv[0]]

from event_order import sort_event_frame
from memory import Config as MemoryConfig
from memory import NodeMemoryModel, fit_standardizer, infer_feature_cols, train_model
from rawdatasets_pre import TemporalBatchGraphSample
from data_pre import DygDataset, UnifiedPadSpec
from model.CensNet import CensNet
from model.Combine import CombinedModel
from model.Transformer import TransformerTargetEdgeClassifier


RELEASE_ROOT = CODE_DIR.parent
DATA_ROOT = Path(os.environ.get("NTDYG_DATA_DIR", RELEASE_ROOT / "data" / "binary"))
SPLIT_ROOT = Path(os.environ.get("NTDYG_SPLIT_DIR", RELEASE_ROOT / "splits" / "ordered"))
DATASET_CONFIG = json.loads((RELEASE_ROOT / "configs" / "model_config.json").read_text())


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_indices(path):
    return pd.read_csv(path, header=None).iloc[:, 0].astype(np.int64).to_numpy()


def load_ordered_splits(dataset_name, csv_path):
    split_dir = SPLIT_ROOT / dataset_name
    summary_path = split_dir / "split_summary.json"
    if summary_path.exists():
        expected_hash = json.loads(summary_path.read_text()).get("source_sha256")
        if expected_hash and sha256(csv_path) != expected_hash:
            raise ValueError("Source CSV does not match these split indices; generate new splits for the current file")
    split_files = {name: split_dir / (name + "_indices.csv") for name in ("train", "validation", "test")}
    indices = {name: read_indices(path) for name, path in split_files.items()}
    source = pd.read_csv(csv_path)
    all_indices = [set(value.tolist()) for value in indices.values()]
    if any(all_indices[i] & all_indices[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("候选数据划分存在交叠，终止训练。")
    if set.union(*all_indices) != set(range(len(source))):
        raise ValueError("候选数据划分未完整覆盖源 CSV，终止训练。")

    required = {"Timestamp_Label", "Label_encoded", "Src IP Label", "Dst IP Label"}
    missing = required - set(source.columns)
    if missing:
        raise ValueError("源 CSV 缺少字段: {}".format(sorted(missing)))

    feature_cols = infer_feature_cols(source, MemoryConfig())
    for column in feature_cols + ["Timestamp_Label", "Label_encoded", "Src IP Label", "Dst IP Label"]:
        source[column] = pd.to_numeric(source[column], errors="raise")
    if not set(source["Label_encoded"].unique()).issubset({0, 1}):
        raise ValueError("当前实验入口只接受 Label_encoded={0,1}。")

    train_raw = source.iloc[indices["train"]].copy()
    train_raw["__source_row"] = indices["train"]
    train_raw = sort_event_frame(train_raw, "Timestamp_Label")
    means, stds = fit_standardizer(train_raw, feature_cols)
    if not np.isfinite(means.to_numpy()).all() or not np.isfinite(stds.to_numpy()).all():
        raise ValueError("训练集标准化统计中存在非有限值。")

    # Encounter-order slots: future suffixes cannot change a prefix ID.
    node2id = {}
    for partition in ("train", "validation", "test"):
        ordered = source.iloc[indices[partition]].assign(__source_row=indices[partition])
        ordered = ordered.sort_values(["Timestamp_Label", "__source_row"], kind="stable")
        for src, dst in ordered[["Src IP Label", "Dst IP Label"]].itertuples(index=False, name=None):
            for entity in (int(src), int(dst)):
                if entity not in node2id:
                    node2id[entity] = len(node2id)
    splits = {}
    for name, row_indices in indices.items():
        frame = source.iloc[row_indices].copy()
        frame["__source_row"] = row_indices
        frame["event_index"] = row_indices.astype(np.int64)
        frame["src_id"] = frame["Src IP Label"].astype(int).map(node2id).astype(np.int64)
        frame["dst_id"] = frame["Dst IP Label"].astype(int).map(node2id).astype(np.int64)
        frame = frame.sort_values(["Timestamp_Label", "__source_row"], kind="stable").reset_index(drop=True)
        frame = frame.astype({column: np.float64 for column in feature_cols})
        frame.loc[:, feature_cols] = (frame[feature_cols] - means) / stds.replace(0, 1.0)
        if not np.isfinite(frame[feature_cols].to_numpy(dtype=np.float64)).all():
            raise ValueError("{} 特征标准化后存在 NaN/Inf。".format(name))
        splits[name] = frame
    return source, indices, splits, feature_cols, node2id


def extend_observed_state(memory, last_update, src, dst):
    """Allocate zero states only when the currently observed event requires them."""
    required = max(src, dst) + 1
    if required > memory.shape[0]:
        extra = required - memory.shape[0]
        memory = torch.cat((memory, memory.new_zeros((extra, memory.shape[1]))), dim=0)
        last_update = torch.cat((last_update, last_update.new_zeros(extra)), dim=0)
    return memory, last_update


@torch.no_grad()
def replay_split(cfg, model, frame, feature_cols, memory_state=None):
    model.eval()
    if memory_state is None:
        memory_matrix, last_update = model.init_memory()
    else:
        memory_matrix, last_update = memory_state
        memory_matrix = memory_matrix.clone()
        last_update = last_update.clone()

    records = []
    logits = []
    source_memories = []
    destination_memories = []
    event_indices = frame["event_index"].to_numpy(dtype=np.int64)
    labels = frame["Label_encoded"].to_numpy(dtype=np.int64)
    timestamps = frame["Timestamp_Label"].to_numpy(dtype=np.float64)
    sources = frame["src_id"].to_numpy(dtype=np.int64)
    destinations = frame["dst_id"].to_numpy(dtype=np.int64)
    feature_matrix = frame[feature_cols].to_numpy(dtype=np.float32, copy=True)
    for event_index, label, timestamp, src_id, dst_id, feature_row in zip(
        event_indices, labels, timestamps, sources, destinations, feature_matrix
    ):
        src_id, dst_id, timestamp = int(src_id), int(dst_id), float(timestamp)
        event_feature = torch.as_tensor(
            feature_row,
            device=cfg.device,
        )
        memory_matrix, last_update = extend_observed_state(
            memory_matrix, last_update, src_id, dst_id)
        src_new, dst_new, _, _, logit = model.compute_updated_states_and_score(
            src_id, dst_id, timestamp, event_feature, memory_matrix, last_update
        )
        memory_matrix[src_id] = src_new
        memory_matrix[dst_id] = dst_new
        last_update[src_id] = timestamp
        last_update[dst_id] = timestamp

        record = {
            "event_index": int(event_index),
            "label": int(label),
            "time": timestamp,
            "src_id": src_id,
            "dst_id": dst_id,
        }
        record.update({column: float(value) for column, value in zip(feature_cols, feature_row)})
        records.append(record)
        logits.append(logit.detach())
        source_memories.append(src_new.detach())
        destination_memories.append(dst_new.detach())

    score_values = (1.0 - torch.sigmoid(torch.stack(logits))).cpu().numpy()
    source_values = torch.stack(source_memories).cpu().numpy()
    destination_values = torch.stack(destination_memories).cpu().numpy()
    for record, score, src_memory, dst_memory in zip(
        records, score_values, source_values, destination_values
    ):
        record["gru_anomaly_score"] = float(score)
        record.update({"src_mem_{}".format(i): float(value) for i, value in enumerate(src_memory)})
        record.update({"dst_mem_{}".format(i): float(value) for i, value in enumerate(dst_memory)})
    return pd.DataFrame(records), (memory_matrix, last_update)


@torch.no_grad()
def replay_memory_scores(cfg, model, frame, feature_cols, memory_state=None):
    """Replay an ordered partition and return anomaly scores without feature dumps."""
    model.eval()
    if memory_state is None:
        memory_matrix, last_update = model.init_memory()
    else:
        memory_matrix, last_update = memory_state
        memory_matrix = memory_matrix.clone()
        last_update = last_update.clone()

    logits = []
    sources = frame["src_id"].to_numpy(dtype=np.int64)
    destinations = frame["dst_id"].to_numpy(dtype=np.int64)
    timestamps = frame[cfg.time_col].to_numpy(dtype=np.float64)
    features = frame[feature_cols].to_numpy(dtype=np.float32, copy=True)
    for src_id, dst_id, timestamp, feature_row in zip(
        sources, destinations, timestamps, features
    ):
        src_id, dst_id, timestamp = int(src_id), int(dst_id), float(timestamp)
        feature = torch.as_tensor(
            feature_row,
            dtype=torch.float32,
            device=cfg.device,
        )
        memory_matrix, last_update = extend_observed_state(
            memory_matrix, last_update, src_id, dst_id)
        src_new, dst_new, _, _, logit = model.compute_updated_states_and_score(
            src_id, dst_id, timestamp, feature, memory_matrix, last_update
        )
        memory_matrix[src_id] = src_new
        memory_matrix[dst_id] = dst_new
        last_update[src_id] = timestamp
        last_update[dst_id] = timestamp
        logits.append(logit.detach())
    scores = (1.0 - torch.sigmoid(torch.stack(logits))).cpu().numpy().astype(np.float64)
    return scores, (memory_matrix, last_update)


def make_graph_data(graph_cfg, events, history, feature_cols):
    sampler = TemporalBatchGraphSample(
        config=graph_cfg,
        target_df=events,
        base_history_df=history,
        edge_feature_cols=feature_cols,
    )
    node, edge, tmat, adj, eadj, mask, target = sampler.get_batch_data()
    return {
        "nodefeatures": node,
        "edgefeatures": edge,
        "masks": np.asarray(mask),
        "labels": events["label"].to_numpy(dtype=np.float32),
        "Tmats": tmat,
        "adjs": adj,
        "eadjs": eadj,
        "event_index": events["event_index"].to_numpy(dtype=np.int64),
        "time": events["time"].to_numpy(dtype=np.float32),
        "target_edge_indices": np.asarray(target, dtype=np.int64),
        "gru_scores": events["gru_anomaly_score"].to_numpy(dtype=np.float32),
    }


def save_graph_data(path, graph_data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump(graph_data, stream, protocol=pickle.HIGHEST_PROTOCOL)


def dynamic_collate(batch):
    batch_size = len(batch)
    nmax = max(item["num_nodes"].item() for item in batch)
    emax = max(item["num_edges"].item() for item in batch)
    node_dim = batch[0]["input_nodes_feature"].shape[1]
    edge_dim = batch[0]["input_edges_feature"].shape[1]
    nodes = torch.zeros((batch_size, nmax, node_dim), dtype=torch.float32)
    edges = torch.zeros((batch_size, emax, edge_dim), dtype=torch.float32)
    tmats = torch.zeros((batch_size, nmax, emax), dtype=torch.float32)
    adjs = torch.zeros((batch_size, nmax, nmax), dtype=torch.float32)
    eadjs = torch.zeros((batch_size, emax, emax), dtype=torch.float32)
    edge_mask = torch.ones((batch_size, emax), dtype=torch.long)
    labels = torch.zeros(batch_size, dtype=torch.float32)
    num_nodes = torch.zeros(batch_size, dtype=torch.long)
    num_edges = torch.zeros(batch_size, dtype=torch.long)
    target = torch.full((batch_size,), -1, dtype=torch.long)

    for i, item in enumerate(batch):
        n, e = item["num_nodes"].item(), item["num_edges"].item()
        nodes[i, :n] = item["input_nodes_feature"]
        edges[i, :e] = item["input_edges_feature"]
        tmats[i, :n, :e] = item["Tmats"]
        adjs[i, :n, :n] = item["adjs"]
        eadjs[i, :e, :e] = item["eadjs"]
        edge_mask[i, :e] = 0
        labels[i] = item["labels"]
        num_nodes[i], num_edges[i] = n, e
        target[i] = item["target_edge_index"]
    return {
        "input_nodes_feature": nodes,
        "input_edges_feature": edges,
        "Tmats": tmats,
        "adjs": adjs,
        "eadjs": eadjs,
        "edge_mask": edge_mask,
        "labels": labels,
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "target_edge_index": target,
    }


def run_model(model, batch, device):
    def dense(tensor):
        return tensor.to_dense() if tensor.is_sparse else tensor
    nodes = dense(batch["input_nodes_feature"].to(device))
    edges = dense(batch["input_edges_feature"].to(device))
    tmats = dense(batch["Tmats"].to(device))
    adjs = dense(batch["adjs"].to(device))
    eadjs = dense(batch["eadjs"].to(device))
    edge_mask = batch["edge_mask"].to(device)
    return model(
        nodes, edges, edges, eadjs, adjs, tmats, edge_mask,
        num_nodes=batch["num_nodes"].to(device),
        num_edges=batch["num_edges"].to(device),
        target_edge_index=batch["target_edge_index"].to(device),
    )[0]


@torch.no_grad()
def score_loader(model, loader, device):
    model.eval()
    scores, labels = [], []
    for batch in loader:
        logits = run_model(model, batch, device)
        scores.append(logits[:, 1].cpu().numpy())
        labels.append(batch["labels"].numpy().astype(np.int64))
    return np.concatenate(scores), np.concatenate(labels)


def choose_threshold(labels, scores):
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    ends = np.r_[np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]), len(scores) - 1]
    if len(ends) == 0:
        return float("inf")
    thresholds = sorted_scores[ends]
    tp = np.cumsum(sorted_labels == 1)[ends]
    fp = np.cumsum(sorted_labels == 0)[ends]
    fn = int(np.sum(labels == 1)) - tp
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(int(np.sum(labels == 1)), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    best = np.flatnonzero(f1 == np.nanmax(f1))
    return float(thresholds[best[0]])


def ranking_metrics(labels, scores):
    positives = labels == 1
    n_pos = int(positives.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("ROC-AUC/AP 要求评估分区同时含正常与异常样本。")

    ranks = rankdata(scores, method="average")
    auc = (ranks[positives].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    end = np.r_[np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]), len(scores) - 1]
    cumulative_tp = np.cumsum(sorted_labels == 1)[end]
    cumulative_fp = np.cumsum(sorted_labels == 0)[end]
    recall = cumulative_tp / n_pos
    precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1)
    previous_recall = np.r_[0.0, recall[:-1]]
    ap = np.sum((recall - previous_recall) * precision)
    return float(auc), float(ap)


def metric_record(labels, scores, threshold):
    pred = scores >= threshold
    tp = int(np.sum((labels == 1) & pred))
    fp = int(np.sum((labels == 0) & pred))
    fn = int(np.sum((labels == 1) & ~pred))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    auc, ap = ranking_metrics(labels, scores)
    return {
        "events": int(len(labels)),
        "anomalies": int(labels.sum()),
        "anomaly_prevalence": float(labels.mean()),
        "roc_auc": auc,
        "average_precision": ap,
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(2 * precision * recall / max(precision + recall, 1e-12)),
        "false_positives": fp,
    }


def ranking_loss(logits, labels):
    normal = logits[labels.long() == 0]
    if normal.shape[0] == 0:
        return logits.sum() * 0.0
    score = normal[:, 1] - normal[:, 0]
    intra = torch.nn.functional.softplus(score + 0.05).mean()
    if score.numel() < 4:
        return intra
    k = max(1, int(score.numel() * 0.20))
    k = min(k, score.numel() // 2)
    if not k:
        return intra
    order = torch.argsort(score.detach())
    pseudo = torch.nn.functional.softplus(-(score[order[-k:]] - score[order[:k]] - 0.10)).mean()
    return intra + 0.20 * pseudo


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.no_grad()
def evaluate_objective(model, loader, device):
    model.eval()
    scores, labels, total, n = [], [], 0.0, 0
    for batch in loader:
        logits = run_model(model, batch, device)
        y = batch["labels"].to(device).long()
        loss = 0.5 * nn.functional.cross_entropy(logits, y) + 0.5 * ranking_loss(logits, y)
        total += float(loss) * len(y)
        n += len(y)
        scores.append(logits[:, 1].cpu().numpy())
        labels.append(batch["labels"].numpy().astype(np.int64))
    return np.concatenate(scores), np.concatenate(labels), total / n


def train_classifier(args, dataset_name, run_dir, loaders, device, config):
    model_config = SimpleNamespace(
        num_classes=2,
        input_dim=config["feature_dim"],
        n_heads=config["heads"],
        drop_out=0.2,
        n_layer=config["layers"],
    )
    gnn = CensNet(17, config["feature_dim"], model_config.drop_out)
    fusion = TransformerTargetEdgeClassifier(model_config, device, hidden_size=config["hidden"])
    model = CombinedModel(
        model_config,
        gnn,
        fusion,
        ablation=args.ablation if args.ablation in {"A2", "A3"} else "full",
        node_memory_dim=16,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    ce = nn.CrossEntropyLoss()
    best_score = -float("inf")
    best_epoch = None
    best_state = None
    stale = 0
    history = []

    for epoch in range(1, args.model_epochs + 1):
        start = time.time()
        model.train()
        total_loss, seen = 0.0, 0
        for batch in loaders["train"]:
            labels = batch["labels"].to(device).long()
            optimizer.zero_grad(set_to_none=True)
            logits = run_model(model, batch, device)
            loss = 0.5 * ce(logits, labels) + 0.5 * ranking_loss(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(labels)
            seen += len(labels)

        # Evaluate both partitions with the same frozen epoch weights and objective.
        _, _, train_eval_loss = evaluate_objective(model, loaders["train"], device)
        val_scores, val_labels, validation_loss = evaluate_objective(model, loaders["validation"], device)
        val_auc, val_ap = ranking_metrics(val_labels, val_scores)
        record = {
            "epoch": epoch,
            "train_loss": total_loss / max(seen, 1),
            "train_eval_loss": train_eval_loss,
            "validation_loss": validation_loss,
            "validation_roc_auc": val_auc,
            "validation_average_precision": val_ap,
            "seconds": time.time() - start,
        }
        history.append(record)
        pd.DataFrame(history).to_csv(run_dir / "epoch_metrics.csv", index=False)
        print(json.dumps(record, ensure_ascii=False), flush=True)

        if val_ap > best_score:
            best_score = val_ap
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break

    if best_state is None:
        raise RuntimeError("未能基于验证集选择分类器 checkpoint。")
    model.load_state_dict(best_state)
    torch.save(best_state, run_dir / "best_validation_checkpoint.pt")
    val_scores, val_labels = score_loader(model, loaders["validation"], device)
    threshold = choose_threshold(val_labels, val_scores)
    pd.DataFrame({"label": val_labels, "anomaly_score_class1_logit": val_scores}).to_csv(
        run_dir / "validation_scores.csv", index=False)
    # The test loader is deliberately evaluated once, after checkpoint and threshold selection.
    test_scores, test_labels = score_loader(model, loaders["test"], device)
    test_metrics = metric_record(test_labels, test_scores, threshold)
    pd.DataFrame({"label": test_labels, "anomaly_score_class1_logit": test_scores}).to_csv(
        run_dir / "test_scores.csv", index=False
    )
    pd.DataFrame(history).to_csv(run_dir / "epoch_metrics.csv", index=False)
    result = {
        "dataset": dataset_name,
        "seed": args.seed,
        "ablation": args.ablation,
        "classifier_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "candidate_split": "ordered_60_20_20_validation_selected",
        "memory_epochs": args.memory_epochs,
        "classifier_epochs_requested": args.model_epochs,
        "best_epoch_by_validation_ap": best_epoch,
        "validation_threshold_selected_by_f1": threshold,
        "test_metrics": test_metrics,
        "test_evaluation_calls": 1,
        "test_scores_note": "Disjoint event-order extension; not wall-clock drift or historical reproduction.",
        "epochs_completed": len(history),
        "memory_retrained": True,
        "entity_universe": "training only; unseen slots initialized on observation",
    }
    (run_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    global DATA_ROOT, SPLIT_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASET_CONFIG), default="IIoTAPT")
    parser.add_argument("--data-dir", type=Path, default=DATA_ROOT)
    parser.add_argument("--split-dir", type=Path, default=SPLIT_ROOT)
    parser.add_argument("--memory-epochs", type=int, default=5)
    parser.add_argument("--memory-patience", type=int, default=3)
    parser.add_argument("--model-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--ordered-classifier", action=argparse.BooleanOptionalAction, default=True,
                        help="Keep classifier minibatches in event order for temporal-baseline comparisons.")
    parser.add_argument("--seed", type=int, default=95545)
    parser.add_argument("--ablation", choices=("full", "A1", "A2", "A3"), default="full")
    parser.add_argument("--reuse-memory-cache", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument(
        "--output-dir", type=Path, default=RELEASE_ROOT / "outputs" / "validation_selected"
    )
    args = parser.parse_args(CLI_ARGS)
    DATA_ROOT, SPLIT_ROOT = args.data_dir.resolve(), args.split_dir.resolve()
    if args.memory_epochs < 1 or args.memory_patience < 1 or args.model_epochs < 1 or args.num_threads < 1:
        parser.error("epoch 数和线程数必须为正整数。")
    torch.set_num_threads(args.num_threads)
    torch.set_num_interop_threads(1)

    dataset_name = args.dataset
    config = DATASET_CONFIG[dataset_name]
    csv_path = DATA_ROOT / (dataset_name + "_re.csv")
    run_dir = args.output_dir / (
        "candidate_60_20_20_{}_{}_seed_{}".format(dataset_name, args.ablation, args.seed)
    )
    if run_dir.exists():
        raise FileExistsError("运行目录已存在，为避免覆盖实验产物而终止: {}".format(run_dir))
    run_dir.mkdir(parents=True)
    set_seed(args.seed)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    source, indices, splits, feature_cols, node2id = load_ordered_splits(dataset_name, csv_path)
    if len(feature_cols) != config["feature_dim"]:
        raise ValueError("{} 边特征维度 {} 与配置 {} 不一致。".format(dataset_name, len(feature_cols), config["feature_dim"]))

    split_audit_path = SPLIT_ROOT / dataset_name / "split_audit.json"
    metadata = {
        "dataset": dataset_name,
        "source_csv": str(csv_path.resolve()),
        "split_dir": str((SPLIT_ROOT / dataset_name).resolve()),
        "source_sha256": sha256(csv_path),
        "split_audit": json.loads(split_audit_path.read_text(encoding="utf-8")),
        "feature_columns": feature_cols,
        "feature_scaler_fit_partition": "train",
        "entity_mapping_scope": "prefix-stable encounter IDs; training-only negative-sampling universe; zero state on first observation",
        "graph_sampling": "k=1; up to 10 most recent incoming historical events per hop",
        "edge_aggregation": "each current edge row aggregates its earlier causal predecessor edge columns",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "seed": args.seed,
        "memory_epochs_requested": args.memory_epochs,
        "memory_early_stopping_patience": args.memory_patience,
        "classifier_epochs_requested": args.model_epochs,
        "classifier_early_stopping_patience": args.patience,
        "classifier_batch_size": config["batch"],
        "classifier_training_order": "event order" if args.ordered_classifier else "shuffled cached causal graphs",
        "ablation": args.ablation,
        "ablation_protocol": {
            "A1": "symmetric endpoint-memory pair encoding for both endpoint updates",
            "A2": "zero node-memory coordinates before graph extraction",
            "A3": "zero graph-context representation before feature fusion",
            "full": "no ablation",
        }[args.ablation],
        "note": "Disjoint event-order evaluation; order keys are not verified wall-clock time.",
    }
    memory_cache_dir = None
    memory_cache_result = None
    if args.reuse_memory_cache is not None:
        memory_cache_dir = args.reuse_memory_cache.resolve()
        required_cache_files = (
            "run_metadata.json",
            "results.json",
            "memory_validation_history.json",
            "memory_model.pt",
            "train_terminal_memory.pt",
            "train_event_features.pkl",
            "validation_event_features.pkl",
            "test_event_features.pkl",
        )
        missing_cache_files = [name for name in required_cache_files if not (memory_cache_dir / name).is_file()]
        if missing_cache_files:
            raise FileNotFoundError(
                "复用记忆缓存缺少文件 {}: {}".format(missing_cache_files, memory_cache_dir)
            )
        cache_metadata = json.loads((memory_cache_dir / "run_metadata.json").read_text(encoding="utf-8"))
        cache_results = json.loads((memory_cache_dir / "results.json").read_text(encoding="utf-8"))
        if cache_metadata.get("dataset") != dataset_name:
            raise ValueError("记忆缓存数据集与当前数据集不一致。")
        if cache_metadata.get("source_sha256") != metadata["source_sha256"]:
            raise ValueError("记忆缓存的源 CSV 哈希与当前源数据不一致。")
        if cache_metadata.get("feature_columns") != feature_cols:
            raise ValueError("记忆缓存的特征列与当前特征列不一致。")
        if cache_metadata.get("split_audit", {}).get("splits") != metadata["split_audit"].get("splits"):
            raise ValueError("记忆缓存的划分计数与当前候选划分不一致。")
        if cache_results.get("seed") != args.seed:
            raise ValueError("记忆缓存随机种子与当前请求不一致。")
        memory_cache_result = cache_results
        metadata["reused_memory_cache"] = str(memory_cache_dir)
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("run_dir:", run_dir, flush=True)
    print("split sizes:", {key: len(value) for key, value in splits.items()}, flush=True)

    memory_cfg = MemoryConfig(
        csv_path=str(csv_path),
        train_idx="",
        test_idx="",
        time_col="Timestamp_Label",
        src_col="Src IP Label",
        dst_col="Dst IP Label",
        label_col="Label_encoded",
        feature_cols=feature_cols,
        memory_dim=16,
        message_dim=config["feature_dim"],
        hidden_dim=32,
        classifier_hidden_dim=32,
        lr=1e-3,
        epochs=args.memory_epochs,
        weight_decay=1e-5,
        device="cpu",
        save_dir=str(run_dir / "memory"),
        seed=args.seed,
        num_negatives=3,
    )
    node_count = int(max(splits["train"].src_id.max(), splits["train"].dst_id.max())) + 1
    metadata["training_entity_count"] = node_count
    metadata["registry_rule"] = "source then destination, at first observation; no learned ID embedding"
    metadata["memory_device"] = "cpu"
    metadata["cached_batches"] = "fixed ordered batches; all tensors unchanged; labels stay on CPU"
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    start = time.time()

    def validate_memory(model, epoch):
        _, train_state = replay_memory_scores(
            memory_cfg, model, splits["train"], feature_cols
        )
        validation_scores, _ = replay_memory_scores(
            memory_cfg, model, splits["validation"], feature_cols, train_state
        )
        _, validation_ap = ranking_metrics(
            splits["validation"]["Label_encoded"].to_numpy(dtype=np.int64),
            validation_scores,
        )
        print(
            json.dumps(
                {"memory_epoch": epoch, "validation_average_precision": validation_ap},
                ensure_ascii=False,
            ),
            flush=True,
        )
        return validation_ap

    if memory_cache_dir is None:
        memory_model = train_model(
            memory_cfg,
            splits["train"],
            feature_cols,
            node_count,
            validation_callback=validate_memory,
            validation_patience=args.memory_patience,
            direction_agnostic_messages=(args.ablation == "A1"),
        )
        memory_history = {
            "best_epoch": memory_model.best_epoch,
            "epochs": memory_model.validation_history,
            "selection_metric": "validation average precision of anomaly score",
        }
        train_events, train_state = replay_split(memory_cfg, memory_model, splits["train"], feature_cols)
        valid_events, valid_state = replay_split(memory_cfg, memory_model, splits["validation"], feature_cols, train_state)
        test_events, _ = replay_split(memory_cfg, memory_model, splits["test"], feature_cols, valid_state)
        memory_seconds = time.time() - start
    else:
        memory_model = NodeMemoryModel(
            num_nodes=node_count,
            feature_dim=config["feature_dim"],
            memory_dim=memory_cfg.memory_dim,
            message_dim=memory_cfg.message_dim,
            hidden_dim=memory_cfg.hidden_dim,
            classifier_hidden_dim=memory_cfg.classifier_hidden_dim,
            device=str(device),
            direction_agnostic_messages=(args.ablation == "A1"),
        )
        memory_model.load_state_dict(
            torch.load(memory_cache_dir / "memory_model.pt", map_location="cpu", weights_only=True)
        )
        memory_history = json.loads(
            (memory_cache_dir / "memory_validation_history.json").read_text(encoding="utf-8")
        )
        memory_model.best_epoch = memory_history.get("best_epoch")
        memory_model.validation_history = memory_history.get("epochs", [])
        train_events = pd.read_pickle(memory_cache_dir / "train_event_features.pkl")
        valid_events = pd.read_pickle(memory_cache_dir / "validation_event_features.pkl")
        test_events = pd.read_pickle(memory_cache_dir / "test_event_features.pkl")
        train_state = torch.load(
            memory_cache_dir / "train_terminal_memory.pt", map_location="cpu", weights_only=True
        )
        memory_history["reused_from"] = str(memory_cache_dir)
        memory_seconds = float(memory_cache_result.get("memory_training_and_replay_seconds", 0.0))
        print("reused memory/cache from:", memory_cache_dir, flush=True)
        print("original memory train/replay seconds:", round(memory_seconds, 2), flush=True)

    (run_dir / "memory_validation_history.json").write_text(
        json.dumps(memory_history, indent=2), encoding="utf-8"
    )
    torch.save(memory_model.state_dict(), run_dir / "memory_model.pt")
    torch.save(
        memory_model.state_dict(),
        run_dir / "best_validation_memory_checkpoint.pt",
    )
    if isinstance(train_state, tuple):
        terminal_state = {"memory": train_state[0].cpu(), "last_update": train_state[1].cpu()}
    else:
        terminal_state = train_state
    torch.save(terminal_state, run_dir / "train_terminal_memory.pt")
    train_events.to_pickle(run_dir / "train_event_features.pkl")
    valid_events.to_pickle(run_dir / "validation_event_features.pkl")
    test_events.to_pickle(run_dir / "test_event_features.pkl")
    print("memory train/replay seconds:", round(memory_seconds, 2), flush=True)

    train_max_node_id = int(max(train_events["src_id"].max(), train_events["dst_id"].max()))
    graph_cfg = SimpleNamespace(
        k_hop=1,
        n_degree=10,
        max_node_id=max(train_max_node_id, 1),
    )
    history = {}
    event_frames = {"train": train_events, "validation": valid_events, "test": test_events}
    for name, event_frame in event_frames.items():
        history_df = None if name == "train" else (
            train_events if name == "validation" else pd.concat([train_events, valid_events], ignore_index=True)
        )
        print("building {} causal graphs ({})".format(name, len(event_frame)), flush=True)
        graph = make_graph_data(graph_cfg, event_frame, history_df, feature_cols)
        path = run_dir / "graphs" / (name + ".pkl")
        save_graph_data(path, graph)
        history[name] = str(path)

    train_path, val_path, test_path = (Path(history[name]) for name in ("train", "validation", "test"))
    pad_spec = UnifiedPadSpec(1, 1, 17, config["feature_dim"])
    datasets = {
        "train": DygDataset(str(train_path), pad_spec),
        "validation": DygDataset(str(val_path), pad_spec),
        "test": DygDataset(str(test_path), pad_spec),
    }
    loaders = {
        name: DataLoader(
            dataset,
            batch_size=config["batch"],
            shuffle=(name == "train" and not args.ordered_classifier),
            num_workers=0,
            collate_fn=dynamic_collate,
            generator=torch.Generator().manual_seed(args.seed),
        )
        for name, dataset in datasets.items()
    }
    if not args.ordered_classifier or args.reuse_memory_cache is not None:
        raise ValueError("This registered experiment requires ordered batches and fresh memory training")
    for partition, loader in list(loaders.items()):
        print("caching fixed batches:", partition, flush=True)
        loaders[partition] = [
            {key: value if key == "labels" else value.to(device) for key, value in batch.items()}
            for batch in loader]
    result = train_classifier(args, dataset_name, run_dir, loaders, device, config)
    result["memory_training_and_replay_seconds"] = memory_seconds
    result["memory_best_epoch_by_validation_ap"] = memory_model.best_epoch
    result["memory_parameter_count"] = sum(
        parameter.numel() for parameter in memory_model.parameters()
    )
    result["parameter_count"] = (
        result["memory_parameter_count"] + result["classifier_parameter_count"]
    )
    (run_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("RESULT:", json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
