#!/usr/bin/env python3
"""Restore the published limited-label settings for added temporal encoders.

The published 1%/5% all-class draws were not retained. We save new, nested,
uniform draws from the full retained source and reuse the retained test indices.
This follows the declared sampling rule but is NOT an exact reconstruction of
historical partitions. Overlap and test-AUC checkpoint selection are recorded.
"""
import argparse
import os
import ast
import json
import random
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from run_legacy_temporal_baseline import (
    CODE, PROJECT, UPSTREAM, CONFIG, MODELS, LegacyEventClassifier, NeighborSampler,
    sha256,
)

SOURCE = PROJECT / "legacy" / "multiclass"
DATA_DIR = Path(os.environ.get("NTDYG_MULTICLASS_DATA_DIR", PROJECT / "data" / "multiclass"))
MULTICLASS_SPLITS = Path(os.environ.get("NTDYG_MULTICLASS_SPLIT_DIR", PROJECT / "splits" / "limited_label"))
SEEDS = (95545, 95546, 95547, 95548)
RATES = (1, 5)
EXCLUDED = {"Src IP Label", "Dst IP Label", "Timestamp_Label", "Label1_encoded", "Label2_encoded"}
MULTICLASS_CONFIG = dict(batch=1024, lr=1e-4, hidden=258, dropout=0.3)
HEADS = dict(CloudDDoS=2, IIoTAPT=4, Darknet=8)
PROTOCOL = "declared_1pct_5pct_restoration_with_retained_test_indices"


def write_json(path, value):
    path = Path(path)
    staged = path.with_suffix(path.suffix + ".writing")
    staged.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    staged.replace(path)


@lru_cache(None)
def original_components():
    """Load only two definitions; never execute an old training/data entrypoint."""
    filename = SOURCE / "utils.py"
    tree = ast.parse(filename.read_text())
    wanted = {"bpr_loss_multiclass_safe", "EarlyStopMonitor"}
    definitions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                   and node.name in wanted]
    if {node.name for node in definitions} != wanted:
        raise RuntimeError("Original multiclass loss / stopping definitions missing")
    namespace = dict(torch=torch, np=np, F=nn.functional)
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(filename), "exec"), namespace)
    return namespace


def loss_function(logits, labels):
    ranking = original_components()["bpr_loss_multiclass_safe"]
    idx = torch.arange(len(labels), device=logits.device)
    return .5 * nn.functional.cross_entropy(logits, labels) + .5 * ranking(logits, labels, idx)


def draws(count, seed):
    """Prespecified whole-source draws; 1% is nested inside 5%, no outcome search."""
    shuffled = np.random.RandomState(seed).permutation(count)
    return {rate: shuffled[:int(count * rate / 100)].copy() for rate in RATES}


def counts(labels, classes):
    return {str(i): int((labels == i).sum()) for i in range(classes)}


def prepare(root):
    root = Path(root).resolve()
    if (root / "protocol.json").exists():
        raise FileExistsError("Prepared experiment already exists; do not overwrite its draws")
    root.mkdir(parents=True, exist_ok=True)
    protocol = dict(
        protocol=PROTOCOL, seeds=list(SEEDS), rates_percent=list(RATES),
        models=list(MODELS), datasets=list(CONFIG), planned_runs=96,
        historical_partition_reproduction_verified=False,
        sampling="uniform without replacement from full retained source; floor(N*rate); nested rates per seed",
        split_provenance="new saved draws following original manuscript 1%/5% rule; historical all-class draws unavailable",
        test_indices="unchanged retained test indices, not the complement of the training draw",
        overlap="retained and counted explicitly; do not describe these as independent test estimates",
        label_column="Label2_encoded; CloudDDoS 26 classes, Darknet 8 categories, IIoTAPT 2 retained labels",
        excluded_features=sorted(EXCLUDED),
        config=dict(**MULTICLASS_CONFIG, heads=HEADS), max_epochs=200,
        training_order="Timestamp_Label ascending in each partition, shuffle=False, as retained multiclass entrypoint",
        optimizer="Adam, lr=0.0001, weight_decay=0 (as invoked)",
        objective="exact original .5 cross_entropy + .5 bpr_loss_multiclass_safe, 5 sampled negative classes",
        selection="highest test ROC-AUC; AP from same checkpoint; original EarlyStopMonitor(15,tolerance=1e-3)",
        table_metrics="CloudDDoS/Darknet: macro OVR AUC and macro AP on raw class logits, as declared in manuscript; IIoTAPT: positive-class AUC and AP",
        auxiliary_metrics="micro OVR AUC, as in a retained generic evaluation branch, saved for audit, never substituted for declared macro table metric",
        metric_repair="multiclass AP uses an explicit N-by-C label indicator; retained vector-label AP call is invalid for a multiclass score matrix",
        head="ordered source embedding + destination embedding + current flow; two-layer GELU head, output width equals retained class count",
        history="10 recent preceding incident events; training history=train, test history=train+test; no labels enter history",
        architecture="same upstream encoders as binary additions; supervised event-classification adaptations",
        upstream_commit="3aacc36b94b8d2d8293d70a74fdf6d39089b4163",
        outcome_targeted_tuning=False, original_ntdyg_rerun=False,
        code_sha256={str(p.relative_to(PROJECT)): sha256(p) for p in [
            Path(__file__).resolve(), CODE / "run_legacy_temporal_baseline.py", SOURCE / "utils.py",
            SOURCE / "option.py", SOURCE / "NTDyG_V1.py", CODE / "cawn_position_encoder.py",
            *[UPSTREAM / "models" / (name + ".py") for name in MODELS],
            UPSTREAM / "models/modules.py", UPSTREAM / "utils/utils.py"]},
        partitions=[],
    )
    for dataset, config in CONFIG.items():
        path = DATA_DIR / (dataset + "_re.csv")
        test_path = MULTICLASS_SPLITS / (dataset + "_test_indx.csv")
        raw = pd.read_csv(path)
        columns = [c for c in raw.columns if c not in EXCLUDED]
        assert len(columns) == config["feature_dim"]
        labels = raw.Label2_encoded.to_numpy(dtype=np.int64)
        classes = int(labels.max()) + 1
        assert np.array_equal(np.unique(labels), np.arange(classes))
        assert classes == dict(CloudDDoS=26, IIoTAPT=2, Darknet=8)[dataset]
        test = pd.read_csv(test_path, header=None).iloc[:, 0].to_numpy(dtype=np.int64)
        assert len(test) == len(np.unique(test)) and test.min() >= 0 and test.max() < len(raw)
        assert np.array_equal(np.unique(labels[test]), np.arange(classes))
        # Match the old global ID mapping, then reserve index zero for padding.
        unique_nodes = pd.Index(raw["Src IP Label"].astype(int).tolist() + raw["Dst IP Label"].astype(int).tolist()).unique()
        mapping = {n: i + 1 for i, n in enumerate(unique_nodes)}
        node_features = np.zeros((len(mapping) + 1, 16), dtype=np.float32)
        node_features[1:, 0] = np.arange(len(mapping), dtype=np.float32) / max(1, len(mapping) - 1)
        source_hash = sha256(path)
        test_hash = sha256(test_path)
        for seed in SEEDS:
            for rate, train in draws(len(raw), seed).items():
                folder = root / "splits" / dataset / ("rate_" + str(rate)) / ("seed_" + str(seed))
                folder.mkdir(parents=True, exist_ok=False)
                pd.Series(train).to_csv(folder / "train_indices.csv", index=False, header=False)
                pd.Series(test).to_csv(folder / "test_indices.csv", index=False, header=False)
                mean = raw.iloc[train][columns].mean()
                std = raw.iloc[train][columns].std().replace(0, 1.)
                # Partition-wise sort is preserved, including duplicate time keys.
                ordered = [raw.iloc[idx].assign(__source_row=idx).sort_values("Timestamp_Label")
                           for idx in (train, test)]
                events = pd.concat(ordered, ignore_index=True)
                features = ((events[columns] - mean) / std).to_numpy(dtype=np.float32)
                if not np.isfinite(features).all():
                    raise FloatingPointError("Nonfinite standardized features")
                times = events.Timestamp_Label.to_numpy(dtype=np.float64)
                shift = max(0., 1. - float(times.min()))
                np.savez(folder / "prepared_arrays.npz", features=features,
                         labels=events.Label2_encoded.to_numpy(dtype=np.int64),
                         src=events["Src IP Label"].astype(int).map(mapping).to_numpy(dtype=np.int64),
                         dst=events["Dst IP Label"].astype(int).map(mapping).to_numpy(dtype=np.int64),
                         times=times + shift, node_features=node_features,
                         source_rows=events.__source_row.to_numpy(dtype=np.int64),
                         scaler_mean=mean.to_numpy(), scaler_std=std.to_numpy())
                info = dict(dataset=dataset, seed=seed, rate_percent=rate, num_classes=classes,
                            source_count=len(raw), source_csv=str(path), source_csv_sha256=source_hash,
                            test_source=str(test_path), test_source_sha256=test_hash,
                            train_count=len(train), test_count=len(test),
                            train_class_counts=counts(labels[train], classes),
                            test_class_counts=counts(labels[test], classes),
                            train_test_overlap=int(np.intersect1d(train, test).size),
                            feature_columns=columns, time_key_padding_shift=shift,
                            arrays_sha256=sha256(folder / "prepared_arrays.npz"),
                            train_indices_sha256=sha256(folder / "train_indices.csv"),
                            test_indices_sha256=sha256(folder / "test_indices.csv"))
                write_json(folder / "split_metadata.json", info)
                protocol["partitions"].append({k: info[k] for k in info if k not in ("feature_columns",)})
                print(json.dumps({k: info[k] for k in ("dataset", "seed", "rate_percent", "train_count", "test_count", "num_classes", "train_test_overlap")}), flush=True)
    write_json(root / "protocol.json", protocol)
    return protocol


def load_data(root, dataset, rate, seed):
    folder = Path(root) / "splits" / dataset / ("rate_" + str(rate)) / ("seed_" + str(seed))
    info = json.loads((folder / "split_metadata.json").read_text())
    with np.load(folder / "prepared_arrays.npz") as arrays:
        data = {name: arrays[name] for name in arrays.files}
    n = info["train_count"]
    length = len(data["labels"])
    data.update(dataset=dataset, num_classes=info["num_classes"], columns=info["feature_columns"],
                split_info=info, split_dir=folder, edge_ids=np.arange(1, length + 1, dtype=np.int64),
                model_config=dict(**MULTICLASS_CONFIG, heads=HEADS[dataset]),
                positions=dict(train=np.arange(n, dtype=np.int64), test=np.arange(n, length, dtype=np.int64)))
    data["edge_features"] = np.vstack([np.zeros((1, len(data["columns"])), np.float32), data["features"]])

    def sampler(positions):
        adjacency = [[] for _ in range(len(data["node_features"]))]
        for i in positions:
            s, d, t, e = int(data["src"][i]), int(data["dst"][i]), float(data["times"][i]), int(data["edge_ids"][i])
            adjacency[s].append((d, e, t))
            if s != d:
                adjacency[d].append((s, e, t))
        return NeighborSampler(adjacency, sample_neighbor_strategy="recent")

    data["samplers"] = dict(train=sampler(data["positions"]["train"]), test=sampler(np.arange(length)))
    return data


def metrics_from_logits(labels, logits):
    classes = logits.shape[1]
    if not np.isfinite(logits).all():
        raise FloatingPointError("Nonfinite logits")
    if not np.array_equal(np.unique(labels), np.arange(classes)):
        raise ValueError("Test set must contain each retained class")
    indicator = np.eye(classes, dtype=np.int8)[labels]
    auc_macro = float(roc_auc_score(indicator, logits, average="macro"))
    ap_macro = float(average_precision_score(indicator, logits, average="macro"))
    auc_micro = float(roc_auc_score(indicator, logits, average="micro"))
    result = dict(roc_auc=auc_macro, average_precision=ap_macro,
                  macro_ovr_auc=auc_macro, macro_ovr_ap=ap_macro, micro_ovr_auc=auc_micro)
    if classes == 2:
        result.update(roc_auc=float(roc_auc_score(labels, logits[:, 1])),
                      average_precision=float(average_precision_score(labels, logits[:, 1])))
    return result


@torch.no_grad()
def evaluate(model, data, batch_size):
    model.eval()
    model.use_partition(data, "test")
    logits = np.concatenate([model.forward_events(data, pos.numpy()).cpu().numpy()
                             for pos in DataLoader(data["positions"]["test"], batch_size=batch_size, shuffle=False)])
    labels = data["labels"][data["positions"]["test"]]
    return metrics_from_logits(labels, logits), logits


def run(args):
    if not 1 <= args.epochs <= 200:
        raise ValueError("Original budget is at most 200 epochs")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    data = load_data(args.output_dir, args.dataset, args.rate, args.seed)
    folder = args.output_dir / args.dataset / ("rate_" + str(args.rate)) / args.model / ("seed_" + str(args.seed))
    folder.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((args.output_dir / "protocol.json").read_text())
    code_paths = [Path(__file__).resolve(), CODE / "run_legacy_temporal_baseline.py", CODE / "cawn_position_encoder.py",
                  SOURCE / "utils.py", SOURCE / "option.py", SOURCE / "NTDyG_V1.py",
                  UPSTREAM / "models" / (args.model + ".py"), UPSTREAM / "models/modules.py", UPSTREAM / "utils/utils.py"]
    for path in code_paths:
        if sha256(path) != protocol["code_sha256"][str(path.relative_to(PROJECT))]:
            raise RuntimeError("Code changed after experiment registration: " + str(path))
    metadata = dict(protocol=PROTOCOL, model=args.model, dataset=args.dataset, seed=args.seed,
                    rate_percent=args.rate, split=data["split_info"], config=data["model_config"],
                    selection=protocol["selection"], objective=protocol["objective"],
                    table_metrics=protocol["table_metrics"], split_dir=str(data["split_dir"]),
                    historical_partition_reproduction_verified=False, validation_partition=None,
                    epochs_requested=args.epochs, device=args.device, torch=torch.__version__,
                    cawn_position_indexing="equivalent vectorized indexing" if args.model == "CAWN" else None,
                    protocol_file_sha256=sha256(args.output_dir / "protocol.json"),
                    code_sha256={str(p.relative_to(PROJECT)): sha256(p) for p in code_paths})
    write_json(folder / "run_metadata.json", metadata)
    model = LegacyEventClassifier(args.model, data, args.device, args.model == "CAWN").to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=MULTICLASS_CONFIG["lr"])
    early = original_components()["EarlyStopMonitor"](max_round=15, higher_better=True)
    loader = DataLoader(data["positions"]["train"], batch_size=MULTICLASS_CONFIG["batch"], shuffle=False)
    best_auc, best_epoch, best_metrics = -np.inf, None, None
    history, started = [], time.time()
    for epoch in range(1, args.epochs + 1):
        tick = time.time()
        model.train()
        model.use_partition(data, "train")
        losses, seen = 0., 0
        for batch in loader:
            pos = batch.numpy()
            optimizer.zero_grad(set_to_none=True)
            logits = model.forward_events(data, pos)
            labels = torch.as_tensor(data["labels"][pos], device=args.device)
            loss = loss_function(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite multiclass loss")
            loss.backward()
            optimizer.step()
            losses += float(loss.detach()) * len(pos)
            seen += len(pos)
        metrics, logits = evaluate(model, data, MULTICLASS_CONFIG["batch"])
        if metrics["roc_auc"] > best_auc:
            best_auc, best_epoch, best_metrics = metrics["roc_auc"], epoch, dict(metrics)
            torch.save(dict(model=model.state_dict(), epoch=epoch, metrics=metrics), folder / "best_test_auc_checkpoint.pt")
            pos = data["positions"]["test"]
            np.savez(folder / "selected_test_scores.npz", source_rows=data["source_rows"][pos],
                     labels=data["labels"][pos], logits=logits)
        record = dict(epoch=epoch, train_loss=losses / seen, **metrics, best_test_auc=best_auc,
                      best_epoch=best_epoch, elapsed_seconds=time.time() - tick)
        history.append(record)
        pd.DataFrame(history).to_csv(folder / "epoch_metrics.csv", index=False)
        write_json(folder / "status.json", dict(status="running", **record))
        print(json.dumps(record), flush=True)
        if early.early_stop_check(best_auc):
            break
    result = dict(status="complete", model=args.model, dataset=args.dataset, seed=args.seed,
                  rate_percent=args.rate, protocol=PROTOCOL, selected_epoch=best_epoch,
                  epochs_completed=len(history), selection="test ROC-AUC", selected_test_metrics=best_metrics,
                  elapsed_seconds=time.time() - started,
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  historical_partition_reproduction_verified=False)
    write_json(folder / "results.json", result)
    write_json(folder / "status.json", result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--dataset", choices=CONFIG)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--rate", type=int, choices=RATES)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    if args.prepare:
        prepare(args.output_dir)
    else:
        if any(getattr(args, key) is None for key in ("dataset", "model", "rate", "seed")):
            parser.error("Training requires dataset, model, rate, and seed")
        run(args)


if __name__ == "__main__":
    main()
