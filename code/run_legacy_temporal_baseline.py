#!/usr/bin/env python3
"""Temporal encoders on the retained 1x legacy NTDyG classifier protocol.

This reproduces the retained protocol, not a verified reconstruction of the
published table. In particular, overlapping indices and test-AUC checkpoint
selection are retained and explicitly recorded. No validation split is added.
"""

import argparse
import os
import hashlib
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader


CODE = Path(__file__).resolve().parent
PROJECT = CODE.parent
LEGACY = PROJECT / "legacy" / "binary"
DATA_DIR = Path(os.environ.get("NTDYG_DATA_DIR", PROJECT / "data" / "binary"))
RETAINED_SPLITS = Path(os.environ.get("NTDYG_RETAINED_SPLIT_DIR", PROJECT / "splits" / "retained"))
UPSTREAM = CODE / "third_party" / "DyGLib"
sys.path.insert(0, str(UPSTREAM))
from models.DyGFormer import DyGFormer
from models.GraphMixer import GraphMixer
from models.TGAT import TGAT
from models.CAWN import CAWN
from utils.utils import NeighborSampler

CONFIG = {
    "CloudDDoS": dict(feature_dim=314, batch=32, lr=5e-4, hidden=256, heads=2),
    "IIoTAPT": dict(feature_dim=52, batch=64, lr=1e-3, hidden=128, heads=4),
    "Darknet": dict(feature_dim=64, batch=64, lr=1e-3, hidden=128, heads=4),
}
MODELS = ("GraphMixer", "DyGFormer", "TGAT", "CAWN")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def import_legacy(name):
    module_name = "ntdyg_retained_legacy_" + name
    if module_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(module_name, LEGACY / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return sys.modules[module_name]


def read_indices(path):
    return pd.read_csv(path, header=None).iloc[:, 0].to_numpy(dtype=np.int64)


def prepare_data(dataset):
    """Call the original preprocessing unchanged, including its ordering."""
    memory = import_legacy("memory")
    cfg = memory.Config()
    cfg.csv_path = str(DATA_DIR / (dataset + "_re.csv"))
    cfg.train_idx = str(RETAINED_SPLITS / (dataset + "_finetune_indx.csv"))
    cfg.test_idx = str(RETAINED_SPLITS / (dataset + "_test_indx.csv"))
    train, test, node_map, _, columns = memory.load_and_prepare_data(cfg)
    raw = pd.read_csv(cfg.csv_path)
    indices = {"train": read_indices(cfg.train_idx), "test": read_indices(cfg.test_idx)}
    source_rows = {}
    for name in ("train", "test"):
        frame = raw.iloc[indices[name]].copy().reset_index(drop=True)
        frame["__source_row"] = indices[name]
        # The original preprocessing sorts each partition independently.
        source_rows[name] = frame.sort_values(cfg.time_col)["__source_row"].to_numpy(dtype=np.int64)
    if len(columns) != CONFIG[dataset]["feature_dim"]:
        raise ValueError("Legacy feature width differs from the dataset configuration")
    if not (train[cfg.label_col].to_numpy() == 0).all():
        raise ValueError("Expected legacy normal-only training events")
    events = pd.concat([train, test], ignore_index=True)
    features = events[columns].to_numpy(dtype=np.float32)
    if not np.isfinite(features).all():
        raise ValueError("Nonfinite features after original preprocessing")
    times = events[cfg.time_col].to_numpy(dtype=np.float64)
    shift = max(0., 1. - float(times.min()))
    times = times + shift  # zero is reserved for DyGLib padding
    src = events.src_id.to_numpy(dtype=np.int64) + 1
    dst = events.dst_id.to_numpy(dtype=np.int64) + 1
    edge_ids = np.arange(1, len(events) + 1, dtype=np.int64)
    nodes = np.zeros((len(node_map) + 1, 16), dtype=np.float32)
    nodes[1:, 0] = np.arange(len(node_map), dtype=np.float32) / max(1, len(node_map) - 1)
    positions = {
        "train": np.arange(len(train), dtype=np.int64),
        "test": np.arange(len(train), len(events), dtype=np.int64),
    }

    def sampler(selected):
        adjacency = [[] for _ in range(len(nodes))]
        for i in selected:
            s, d, e, t = int(src[i]), int(dst[i]), int(edge_ids[i]), float(times[i])
            adjacency[s].append((d, e, t))
            if s != d:
                adjacency[d].append((s, e, t))
        return NeighborSampler(adjacency, sample_neighbor_strategy="recent")

    means, stds = memory.fit_standardizer(raw.iloc[indices["train"]], columns)
    return dict(
        csv=Path(cfg.csv_path), train_path=Path(cfg.train_idx), test_path=Path(cfg.test_idx),
        source_count=len(raw), indices=indices, source_rows=np.concatenate(list(source_rows.values())),
        positions=positions, columns=columns, features=features,
        node_features=nodes, edge_features=np.vstack([np.zeros((1, len(columns)), np.float32), features]),
        src=src, dst=dst, times=times, edge_ids=edge_ids,
        labels=events[cfg.label_col].to_numpy(dtype=np.int64), shift=shift,
        samplers={"train": sampler(positions["train"]), "test": sampler(np.arange(len(events)))},
        scaler_mean=means.to_numpy(dtype=float), scaler_std=stds.to_numpy(dtype=float),
    )


class LegacyEventClassifier(nn.Module):
    """Unmodified stateless upstream encoder plus a directed flow-event head."""

    def __init__(self, name, data, device="cpu", cawn_indexed_position=False):
        super().__init__()
        self.name = name
        self.device_name = str(device)
        self.history_size = 10
        dataset = data["dataset"]
        config = data.get("model_config", CONFIG[dataset])
        dropout = config.get("dropout", 0.2)
        common = dict(node_raw_features=data["node_features"], edge_raw_features=data["edge_features"],
                      neighbor_sampler=data["samplers"]["train"], time_feat_dim=16,
                      dropout=dropout, device=str(device))
        if name == "GraphMixer":
            self.encoder = GraphMixer(**common, num_tokens=10, num_layers=2)
        elif name == "DyGFormer":
            self.encoder = DyGFormer(**common, channel_embedding_dim=16, patch_size=1,
                                    num_layers=2, num_heads=4, max_input_sequence_length=11)
        elif name == "TGAT":
            self.encoder = TGAT(**common, num_layers=2, num_heads=config["heads"])
        elif name == "CAWN":
            self.encoder = CAWN(**common, position_feat_dim=16,
                                walk_length=2, num_walk_heads=8)
            if cawn_indexed_position:
                from cawn_position_encoder import IndexedPositionEncoder
                # Preserve the initialized parameters and RNG state exactly.
                self.encoder.position_encoder.__class__ = IndexedPositionEncoder
        else:
            raise ValueError(name)
        self.head = nn.Sequential(nn.Linear(32 + len(data["columns"]), config["hidden"]), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(config["hidden"], data.get("num_classes", 2)))

    def use_partition(self, data, name):
        self.encoder.set_neighbor_sampler(data["samplers"][name])

    def forward_events(self, data, positions):
        kwargs = dict(src_node_ids=data["src"][positions], dst_node_ids=data["dst"][positions],
                      node_interact_times=data["times"][positions])
        if self.name in ("GraphMixer", "TGAT", "CAWN"):
            kwargs["num_neighbors"] = self.history_size
        if self.name == "GraphMixer":
            kwargs["time_gap"] = self.history_size
        src, dst = self.encoder.compute_src_dst_node_temporal_embeddings(**kwargs)
        flow = torch.as_tensor(data["features"][positions], device=self.device_name)
        return self.head(torch.cat([src, dst, flow], dim=1))


def loss_function(logits, labels):
    legacy = import_legacy("utils")
    return .5 * nn.functional.cross_entropy(logits, labels) + .5 * legacy.bpr_loss_binary_normal_only(
        logits, labels, torch.arange(len(labels), device=logits.device),
        intra_margin=.05, pseudo_margin=.10, lambda_pseudo=.20, hard_ratio=.20)


@torch.no_grad()
def evaluate(model, data, batch_size):
    model.eval()
    model.use_partition(data, "test")
    outputs = []
    for pos in DataLoader(data["positions"]["test"], batch_size=batch_size, shuffle=False):
        logits = model.forward_events(data, pos.numpy())
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Nonfinite legacy evaluation logits")
        outputs.append(logits.cpu().numpy())
    logits = np.concatenate(outputs)
    y = data["labels"][data["positions"]["test"]]
    return dict(roc_auc=float(roc_auc_score(y, logits[:, 1])),
                average_precision=float(average_precision_score(y, logits[:, 1]))), logits


def run(args):
    if args.epochs < 1 or args.epochs > 200:
        raise ValueError("Legacy epoch ceiling is 200")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    data = prepare_data(args.dataset)
    data["dataset"] = args.dataset
    config = CONFIG[args.dataset]
    run_dir = args.output_dir / args.dataset / args.model / ("seed_" + str(args.seed))
    run_dir.mkdir(parents=True, exist_ok=False)
    metadata = dict(
        status="prepared", model=args.model, dataset=args.dataset, seed=args.seed,
        protocol="retained_1x_legacy", historical_table_reproduction_verified=False,
        source_csv=str(data["csv"]), source_csv_sha256=sha256(data["csv"]),
        train_index_sha256=sha256(data["train_path"]), test_index_sha256=sha256(data["test_path"]),
        train_index_file=str(data["train_path"]), test_index_file=str(data["test_path"]),
        source_count=data["source_count"], feature_columns=data["columns"],
        partition_counts={name: {"events": len(pos), "class_0": int((data["labels"][pos] == 0).sum()),
                                 "class_1": int((data["labels"][pos] == 1).sum())}
                          for name, pos in data["positions"].items()},
        train_test_overlap=int(np.intersect1d(data["indices"]["train"], data["indices"]["test"]).size),
        validation_partition=None, epochs_requested=args.epochs, batch_size=config["batch"],
        learning_rate=config["lr"], optimizer="Adam; weight_decay=0, as invoked by the old entry point",
        training_order="shuffled target events; historical context queried independently for each target",
        feature_scaling="unchanged original memory.load_and_prepare_data; train-fitted mean and sample SD",
        model_selection="strictly highest test ROC-AUC; AP from that same checkpoint",
        early_stopping="unchanged original EarlyStopMonitor(15, tolerance=1e-3), applied to best test AUC",
        loss="unchanged original .5 CE + .5 bpr_loss_binary_normal_only(.05,.10,.20,.20)",
        score="raw class-1 logit, unchanged from original direct-output evaluation",
        scoring_branch="direct neural classifier only; no Isolation Forest or post-hoc score reversal",
        model_adaptation="upstream encoder plus supervised ordered-endpoint/current-flow head; no upstream link-prediction replication claim",
        cawn_position_indexing="equivalent vectorized NumPy indexing" if args.cawn_indexed_position else "native upstream",
        history="10 recent preceding incident events per endpoint; training uses train history only, test uses train+test history",
        time_key_padding_shift=data["shift"], outcome_targeted_tuning=False,
        device=args.device, torch=torch.__version__,
        code_sha256={str(p.relative_to(PROJECT)): sha256(p) for p in
                     [Path(__file__).resolve(), LEGACY/"memory.py", LEGACY/"utils.py", LEGACY/"NTDyG_V1.py", LEGACY/"option.py",
                      UPSTREAM/"models"/(args.model+".py"), UPSTREAM/"models/modules.py", UPSTREAM/"utils/utils.py"]
                     + ([CODE/"cawn_position_encoder.py"] if args.cawn_indexed_position else [])},
    )
    np.savez(run_dir/"scaler.npz", mean=data["scaler_mean"], std=data["scaler_std"])
    (run_dir/"run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+"\n")
    if args.dry_run:
        print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
        return
    model = LegacyEventClassifier(args.model, data, args.device, args.cawn_indexed_position).to(args.device)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=config["lr"])
    early = import_legacy("utils").EarlyStopMonitor(max_round=15, higher_better=True)
    loader = DataLoader(data["positions"]["train"], batch_size=config["batch"], shuffle=True, num_workers=0)
    best_auc, best_metrics, best_epoch, history = -np.inf, None, None, []
    started = time.time()
    for epoch in range(1, args.epochs+1):
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
                raise FloatingPointError("Nonfinite legacy training loss")
            loss.backward()
            optimizer.step()
            losses += float(loss.detach()) * len(pos)
            seen += len(pos)
        metrics, logits = evaluate(model, data, config["batch"])
        if metrics["roc_auc"] > best_auc:
            best_auc, best_metrics, best_epoch = metrics["roc_auc"], dict(metrics), epoch
            torch.save(dict(model=model.state_dict(), epoch=epoch, metrics=metrics), run_dir/"best_test_auc_checkpoint.pt")
            pos = data["positions"]["test"]
            pd.DataFrame(dict(source_row=data["source_rows"][pos], label=data["labels"][pos],
                              logit_0=logits[:, 0], score=logits[:, 1])).to_csv(run_dir/"selected_test_scores.csv", index=False)
        record = dict(epoch=epoch, train_loss=losses/seen, **metrics,
                      best_test_auc=best_auc, best_epoch=best_epoch, elapsed_seconds=time.time()-tick)
        history.append(record)
        pd.DataFrame(history).to_csv(run_dir/"epoch_metrics.csv", index=False)
        (run_dir/"status.json").write_text(json.dumps(dict(status="running", **record), indent=2)+"\n")
        print(json.dumps(record), flush=True)
        if early.early_stop_check(best_auc):
            break
    results = dict(status="complete", model=args.model, dataset=args.dataset, seed=args.seed,
                   protocol="retained_1x_legacy", selected_epoch=best_epoch, epochs_completed=len(history),
                   selection="test ROC-AUC", selected_test_metrics=best_metrics,
                   elapsed_seconds=time.time()-started, trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                   historical_table_reproduction_verified=False)
    (run_dir/"results.json").write_text(json.dumps(results, indent=2)+"\n")
    (run_dir/"status.json").write_text(json.dumps(results, indent=2)+"\n")
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--dataset", choices=CONFIG, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cawn-indexed-position", action="store_true",
                        help="Use output/gradient-equivalent vectorized CAWN count indexing")
    run(parser.parse_args())
