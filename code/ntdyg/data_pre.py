import os
import pickle
from typing import Optional, Dict, Any, List

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset, DataLoader

from option import args


def _to_float_tensor(x) -> torch.Tensor:
    """
    将输入统一转换成 float32 的 torch.Tensor。
    支持:
    - numpy.ndarray
    - scipy.sparse matrix
    - torch.Tensor
    - Python list
    """
    if isinstance(x, torch.Tensor):
        if x.is_sparse:
            x = x.to_dense()
        return x.to(dtype=torch.float32)

    if sp.issparse(x):
        return torch.tensor(x.toarray(), dtype=torch.float32)

    if isinstance(x, np.ndarray):
        return torch.tensor(x, dtype=torch.float32)

    return torch.tensor(np.array(x), dtype=torch.float32)


def _ensure_2d_tensor(x) -> torch.Tensor:
    """
    保证节点/边特征是二维张量:
    - [N] -> [N, 1]
    - [N, D] 保持不变
    """
    x = _to_float_tensor(x)
    if x.ndim == 1:
        x = x.unsqueeze(-1)
    return x


class UnifiedPadSpec:
    """
    保存同一数据集 train/test 共同使用的统一 padding 规格。
    """
    def __init__(
        self,
        max_num_nodes: int,
        max_num_edges: int,
        node_feat_dim: int,
        edge_feat_dim: int
    ):
        self.max_num_nodes = int(max_num_nodes)
        self.max_num_edges = int(max_num_edges)
        self.node_feat_dim = int(node_feat_dim)
        self.edge_feat_dim = int(edge_feat_dim)

    def __repr__(self):
        return (
            f"UnifiedPadSpec("
            f"max_num_nodes={self.max_num_nodes}, "
            f"max_num_edges={self.max_num_edges}, "
            f"node_feat_dim={self.node_feat_dim}, "
            f"edge_feat_dim={self.edge_feat_dim})"
        )


def infer_pad_spec_from_pkl(train_pkl_path: str, test_pkl_path: str) -> UnifiedPadSpec:
    """
    从 train/test pkl 共同统计统一规格，保证训练与测试输入格式一致。
    """
    if not os.path.isfile(train_pkl_path):
        raise FileNotFoundError(f"找不到训练 pkl 文件: {train_pkl_path}")
    if not os.path.isfile(test_pkl_path):
        raise FileNotFoundError(f"找不到测试 pkl 文件: {test_pkl_path}")

    with open(train_pkl_path, "rb") as f:
        train_data = pickle.load(f)
    with open(test_pkl_path, "rb") as f:
        test_data = pickle.load(f)

    for key in ["nodefeatures", "edgefeatures"]:
        if key not in train_data:
            raise ValueError(f"训练 pkl 缺少字段: {key}")
        if key not in test_data:
            raise ValueError(f"测试 pkl 缺少字段: {key}")

    train_nodefeatures = train_data["nodefeatures"]
    train_edgefeatures = train_data["edgefeatures"]
    test_nodefeatures = test_data["nodefeatures"]
    test_edgefeatures = test_data["edgefeatures"]

    all_nodefeatures = list(train_nodefeatures) + list(test_nodefeatures)
    all_edgefeatures = list(train_edgefeatures) + list(test_edgefeatures)

    if len(all_nodefeatures) == 0 or len(all_edgefeatures) == 0:
        raise ValueError("train/test pkl 中 nodefeatures 或 edgefeatures 为空。")

    all_node_shapes = [_ensure_2d_tensor(x).shape for x in all_nodefeatures]
    all_edge_shapes = [_ensure_2d_tensor(x).shape for x in all_edgefeatures]

    # 检查所有样本的特征维度是否一致
    node_feat_dims = {shape[1] for shape in all_node_shapes}
    edge_feat_dims = {shape[1] for shape in all_edge_shapes}

    if len(node_feat_dims) != 1:
        raise ValueError(f"节点特征维度不一致: {node_feat_dims}")
    if len(edge_feat_dims) != 1:
        raise ValueError(f"边特征维度不一致: {edge_feat_dims}")

    max_num_nodes = max(shape[0] for shape in all_node_shapes)
    max_num_edges = max(shape[0] for shape in all_edge_shapes)
    node_feat_dim = next(iter(node_feat_dims))
    edge_feat_dim = next(iter(edge_feat_dims))

    return UnifiedPadSpec(
        max_num_nodes=max_num_nodes,
        max_num_edges=max_num_edges,
        node_feat_dim=node_feat_dim,
        edge_feat_dim=edge_feat_dim
    )


def _get_train_test_pkl_paths(config):
    train_pkl_path = os.path.join(
        config.output_dir,
        config.data_set,
        f"{config.data_set}_train_{config.extype}.pkl"
    )
    test_pkl_path = os.path.join(
        config.output_dir,
        config.data_set,
        f"{config.data_set}_test_{config.extype}.pkl"
    )
    return train_pkl_path, test_pkl_path


class Collate:
    """
    兼容旧代码的 Collate 封装。
    """
    def __init__(self, config):
        self.config = config
        train_pkl_path, test_pkl_path = _get_train_test_pkl_paths(config)
        self.pad_spec = infer_pad_spec_from_pkl(train_pkl_path, test_pkl_path)
        self._collator = UnifiedCollate(self.pad_spec)

    def dyg_collate_fn(self, batch):
        return self._collator.dyg_collate_fn(batch)


class DygDataset(Dataset):
    """
    读取预处理好的动态图 pkl 文件，并封装成统一格式的 Dataset。

    注意：
    - train/test 共用同一个 pad_spec
    - Dataset 返回单样本的原始张量
    - 具体 padding 在 collate_fn 中按统一规格完成
    """
    def __init__(
        self,
        pkl_path: str,
        pad_spec: UnifiedPadSpec
    ):
        super().__init__()

        if not os.path.isfile(pkl_path):
            raise FileNotFoundError(f"找不到 pkl 文件: {pkl_path}")

        self.pkl_path = pkl_path
        self.pad_spec = pad_spec

        with open(self.pkl_path, "rb") as f:
            data = pickle.load(f)

        required_keys = ["nodefeatures", "edgefeatures", "Tmats", "adjs", "eadjs", "labels"]
        missing = [k for k in required_keys if k not in data]
        if missing:
            raise ValueError(f"pkl 文件缺少必要字段: {missing}")

        self.nodefeatures = data["nodefeatures"]
        self.edgefeatures = data["edgefeatures"]
        self.Tmats = data["Tmats"]
        self.adjs = data["adjs"]
        self.eadjs = data["eadjs"]

        self.labels = torch.tensor(np.array(data["labels"]), dtype=torch.float32)
         # 新增：GRU 异常分数
        if "gru_scores" in data:
            self.gru_scores = torch.tensor(np.array(data["gru_scores"]), dtype=torch.float32)
        else:
            self.gru_scores = torch.zeros(len(self.labels), dtype=torch.float32)

        if "event_index" in data:
            self.event_index = torch.tensor(np.array(data["event_index"]), dtype=torch.long)
        else:
            self.event_index = torch.arange(len(self.labels), dtype=torch.long)

        if "time" in data:
            self.time = torch.tensor(np.array(data["time"]), dtype=torch.float32)
        elif "Timestamp_Label" in data:
            self.time = torch.tensor(np.array(data["Timestamp_Label"]), dtype=torch.float32)
        else:
            self.time = torch.arange(len(self.labels), dtype=torch.float32)

        # 新增：目标事件边索引
        if "target_edge_indices" in data:
            self.target_edge_indices = torch.tensor(
                np.array(data["target_edge_indices"]),
                dtype=torch.long
            )
        else:
            # 如果旧 pkl 没有这个字段，给默认值 -1
            self.target_edge_indices = torch.full(
                (len(self.labels),),
                fill_value=-1,
                dtype=torch.long
            )

        n = len(self.labels)
        if not (
            len(self.nodefeatures) == len(self.edgefeatures) == len(self.Tmats) ==
            len(self.adjs) == len(self.eadjs) == len(self.event_index) ==
            len(self.time) == len(self.target_edge_indices) ==
            len(self.gru_scores) == n
        ):
            raise ValueError("pkl 中各字段样本数不一致，请检查预处理结果。")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        node_feat = _ensure_2d_tensor(self.nodefeatures[idx])     # [N, Dn]
        edge_feat = _ensure_2d_tensor(self.edgefeatures[idx])     # [E, De]
        Tmat = _to_float_tensor(self.Tmats[idx])                  # [N, E]
        adj = _to_float_tensor(self.adjs[idx])                    # [N, N]
        eadj = _to_float_tensor(self.eadjs[idx])                  # [E, E]

        num_nodes = node_feat.shape[0]
        num_edges = edge_feat.shape[0]
        target_edge_index = int(self.target_edge_indices[idx].item())

        # 一致性检查
        if node_feat.shape[1] != self.pad_spec.node_feat_dim:
            raise ValueError(
                f"样本 {idx} 节点特征维度错误: {node_feat.shape[1]} != {self.pad_spec.node_feat_dim}"
            )
        if edge_feat.shape[1] != self.pad_spec.edge_feat_dim:
            raise ValueError(
                f"样本 {idx} 边特征维度错误: {edge_feat.shape[1]} != {self.pad_spec.edge_feat_dim}"
            )

        if Tmat.shape[0] != num_nodes or Tmat.shape[1] != num_edges:
            raise ValueError(
                f"样本 {idx} 的 Tmat 维度错误: "
                f"Tmat={tuple(Tmat.shape)}, num_nodes={num_nodes}, num_edges={num_edges}"
            )

        if adj.shape[0] != num_nodes or adj.shape[1] != num_nodes:
            raise ValueError(
                f"样本 {idx} 的 adj 维度错误: "
                f"adj={tuple(adj.shape)}, num_nodes={num_nodes}"
            )

        if eadj.shape[0] != num_edges or eadj.shape[1] != num_edges:
            raise ValueError(
                f"样本 {idx} 的 eadj 维度错误: "
                f"eadj={tuple(eadj.shape)}, num_edges={num_edges}"
            )

        # 新增：目标事件边位置检查
        if target_edge_index != -1:
            if target_edge_index < 0 or target_edge_index >= num_edges:
                raise ValueError(
                    f"样本 {idx} 的 target_edge_index 越界: "
                    f"{target_edge_index}, num_edges={num_edges}"
                )

        return {
            "input_nodes_feature": node_feat,
            "input_edges_feature": edge_feat,
            "Tmats": Tmat,
            "adjs": adj,
            "eadjs": eadj,
            "labels": self.labels[idx],
            "event_index": self.event_index[idx],
            "time": self.time[idx],
            "num_nodes": torch.tensor(num_nodes, dtype=torch.long),
            "num_edges": torch.tensor(num_edges, dtype=torch.long),
            "target_edge_index": torch.tensor(target_edge_index, dtype=torch.long),
            "gru_score": self.gru_scores[idx],
        }


class UnifiedCollate:
    """
    使用全局统一规格进行 padding。
    训练和测试集都按同一个 pad_spec 输出完全一致的格式。
    """
    def __init__(self, pad_spec: UnifiedPadSpec):
        self.pad_spec = pad_spec

    def dyg_collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch_size = len(batch)

        Nmax = self.pad_spec.max_num_nodes
        Emax = self.pad_spec.max_num_edges
        Dn = self.pad_spec.node_feat_dim
        De = self.pad_spec.edge_feat_dim

        padded_node_features = torch.zeros((batch_size, Nmax, Dn), dtype=torch.float32)
        padded_edge_features = torch.zeros((batch_size, Emax, De), dtype=torch.float32)

        padded_Tmats = torch.zeros((batch_size, Nmax, Emax), dtype=torch.float32)
        padded_adjs = torch.zeros((batch_size, Nmax, Nmax), dtype=torch.float32)
        padded_eadjs = torch.zeros((batch_size, Emax, Emax), dtype=torch.float32)

        # 0 有效，1 padding
        node_mask = torch.ones((batch_size, Nmax), dtype=torch.long)
        edge_mask = torch.ones((batch_size, Emax), dtype=torch.long)

        labels = torch.zeros(batch_size, dtype=torch.float32)
        event_index = torch.zeros(batch_size, dtype=torch.long)
        time = torch.zeros(batch_size, dtype=torch.float32)
        num_nodes = torch.zeros(batch_size, dtype=torch.long)
        num_edges = torch.zeros(batch_size, dtype=torch.long)

        # 新增：目标事件边在 padded edgefeatures 中的位置
        # 若原始 target_edge_index 为 -1，则保持 -1
        target_edge_index = torch.full((batch_size,), fill_value=-1, dtype=torch.long)
        gru_score = torch.zeros(batch_size, dtype=torch.float32)

        for i, item in enumerate(batch):
            n = item["num_nodes"].item()
            e = item["num_edges"].item()
            tei = item["target_edge_index"].item()

            if n > Nmax:
                raise ValueError(f"样本节点数 {n} 超过统一上限 {Nmax}")
            if e > Emax:
                raise ValueError(f"样本边数 {e} 超过统一上限 {Emax}")

            padded_node_features[i, :n, :] = item["input_nodes_feature"]
            padded_edge_features[i, :e, :] = item["input_edges_feature"]

            padded_Tmats[i, :n, :e] = item["Tmats"]
            padded_adjs[i, :n, :n] = item["adjs"]
            padded_eadjs[i, :e, :e] = item["eadjs"]

            node_mask[i, :n] = 0
            edge_mask[i, :e] = 0

            labels[i] = item["labels"]
            event_index[i] = item["event_index"]
            time[i] = item["time"]
            num_nodes[i] = n
            num_edges[i] = e
            gru_score[i] = item["gru_score"]

            if tei != -1:
                if tei < 0 or tei >= e:
                    raise ValueError(
                        f"batch 内样本 {i} 的 target_edge_index 越界: {tei}, num_edges={e}"
                    )
                target_edge_index[i] = tei

        return {
            "input_nodes_feature": padded_node_features,   # [B, Nmax, Dn]
            "input_edges_feature": padded_edge_features,   # [B, Emax, De]
            "Tmats": padded_Tmats,                         # [B, Nmax, Emax]
            "adjs": padded_adjs,                           # [B, Nmax, Nmax]
            "eadjs": padded_eadjs,                         # [B, Emax, Emax]
            "node_mask": node_mask,                        # [B, Nmax]
            "edge_mask": edge_mask,                        # [B, Emax]
            "labels": labels,                              # [B]
            "event_index": event_index,                    # [B]
            "time": time,                                  # [B]
            "num_nodes": num_nodes,                        # [B]
            "num_edges": num_edges,                        # [B]
            "target_edge_index": target_edge_index,        # [B]
            "gru_score": gru_score,
        }


class DygDataModule:
    """
    管理同一数据集 train/test 的统一规格与 DataLoader。
    """
    def __init__(self, config):
        self.config = config

        self.train_pkl_path = os.path.join(
            config.output_dir,
            config.data_set,
            f"{config.data_set}_train_{config.extype}.pkl"
        )
        self.test_pkl_path = os.path.join(
            config.output_dir,
            config.data_set,
            f"{config.data_set}_test_{config.extype}.pkl"
        )

        self.pad_spec = infer_pad_spec_from_pkl(
            self.train_pkl_path,
            self.test_pkl_path
        )

        self.train_dataset = DygDataset(
            pkl_path=self.train_pkl_path,
            pad_spec=self.pad_spec
        )
        self.test_dataset = DygDataset(
            pkl_path=self.test_pkl_path,
            pad_spec=self.pad_spec
        )

        self.collate = UnifiedCollate(self.pad_spec)

    def train_dataloader(self, batch_size: Optional[int] = None, shuffle: bool = True, num_workers: int = 0):
        if batch_size is None:
            batch_size = getattr(self.config, "batch_size", 8)

        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collate.dyg_collate_fn,
            drop_last=False
        )

    def test_dataloader(self, batch_size: Optional[int] = None, shuffle: bool = False, num_workers: int = 0):
        if batch_size is None:
            batch_size = getattr(self.config, "batch_size", 8)

        return DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collate.dyg_collate_fn,
            drop_last=False
        )


if __name__ == "__main__":
    config = args

    if not hasattr(config, "k_hop"):
        config.k_hop = 1
    if not hasattr(config, "batch_size"):
        config.batch_size = 4

    dm = DygDataModule(config)

    print("统一规格:")
    print(dm.pad_spec)

    print("训练集样本数:", len(dm.train_dataset))
    print("测试集样本数:", len(dm.test_dataset))

    train_loader = dm.train_dataloader(batch_size=config.batch_size, shuffle=True, num_workers=0)
    test_loader = dm.test_dataloader(batch_size=config.batch_size, shuffle=False, num_workers=0)

    train_batch = next(iter(train_loader))
    test_batch = next(iter(test_loader))

    print("\n===== 训练 batch 形状 =====")
    print("input_nodes_feature:", train_batch["input_nodes_feature"].shape)
    print("input_edges_feature:", train_batch["input_edges_feature"].shape)
    print("Tmats:", train_batch["Tmats"].shape)
    print("adjs:", train_batch["adjs"].shape)
    print("eadjs:", train_batch["eadjs"].shape)
    print("node_mask:", train_batch["node_mask"].shape)
    print("edge_mask:", train_batch["edge_mask"].shape)
    print("labels:", train_batch["labels"].shape)
    print("event_index:", train_batch["event_index"].shape)
    print("time:", train_batch["time"].shape)
    print("target_edge_index:", train_batch["target_edge_index"].shape)
    print("gru_score:", train_batch["gru_score"].shape)

    print("\n===== 测试 batch 形状 =====")
    print("input_nodes_feature:", test_batch["input_nodes_feature"].shape)
    print("input_edges_feature:", test_batch["input_edges_feature"].shape)
    print("Tmats:", test_batch["Tmats"].shape)
    print("adjs:", test_batch["adjs"].shape)
    print("eadjs:", test_batch["eadjs"].shape)
    print("node_mask:", test_batch["node_mask"].shape)
    print("edge_mask:", test_batch["edge_mask"].shape)
    print("labels:", test_batch["labels"].shape)
    print("event_index:", test_batch["event_index"].shape)
    print("time:", test_batch["time"].shape)
    print("target_edge_index:", test_batch["target_edge_index"].shape)
    print("gru_score:", test_batch["gru_score"].shape)