import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["LOKY_MAX_CPU_COUNT"] = "8"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from tqdm import tqdm
from memory import NodeMemoryModel, load_and_prepare_data
from event_order import sort_event_frame
import pickle
from option import args
import pandas as pd
import torch
import numpy as np
import scipy.sparse as sp
import re
from dataclasses import dataclass
from typing import List, Tuple
from bisect import bisect_right

@dataclass
class Config:
    csv_path: str = "dataset/IIoTAPT_re.csv"
    train_idx: str = "dataset/IIoTAPT_finetune_indx.csv"
    test_idx: str = "dataset/IIoTAPT_test_indx.csv"

    time_col: str = "Timestamp_Label"
    src_col: str = "Src IP Label"
    dst_col: str = "Dst IP Label"
    label_col: str = "Label_encoded"
    feature_cols: List[str] = None

    memory_dim: int = 16
    message_dim: int = 52
    hidden_dim: int = 32
    classifier_hidden_dim: int = 32

    lr: float = 1e-3
    epochs: int = 10
    weight_decay: float = 1e-5
    device: str = "cuda:{}".format(0) if torch.cuda.is_available() else "cpu"

    save_dir: str = "outputs/IIoTAPT"
    record_every_event: bool = True
    seed: int = 42
    num_negatives: int = 3


def build_cfg_for_dataset(dataset_name: str, output_dir: str) -> Config:
    if dataset_name == "IIoTAPT":
        return Config(
            csv_path="dataset/IIoTAPT_re.csv",
            train_idx="dataset/IIoTAPT_finetune_indx.csv",
            test_idx="dataset/IIoTAPT_test_indx.csv",
            time_col="Timestamp_Label",
            src_col="Src IP Label",
            dst_col="Dst IP Label",
            label_col="Label_encoded",
            memory_dim=16,
            message_dim=52,
            hidden_dim=32,
            classifier_hidden_dim=32,
            lr=1e-3,
            epochs=200,
            save_dir=os.path.join(output_dir, "IIoTAPT"),
        )
    elif dataset_name == "CloudDDoS":
        return Config(
            csv_path="dataset/CloudDDoS_re.csv",
            train_idx="dataset/CloudDDoS_finetune_indx.csv",
            test_idx="dataset/CloudDDoS_test_indx.csv",
            time_col="Timestamp_Label",
            src_col="Src IP Label",
            dst_col="Dst IP Label",
            label_col="Label_encoded",
            memory_dim=16,
            message_dim=314,
            hidden_dim=32,
            classifier_hidden_dim=32,
            lr=1e-3,
            epochs=200,
            save_dir=os.path.join(output_dir, "CloudDDoS"),
        )
    elif dataset_name == "Darknet":
        return Config(
            csv_path="dataset/Darknet_re.csv",
            train_idx="dataset/Darknet_finetune_indx.csv",
            test_idx="dataset/Darknet_test_indx.csv",
            time_col="Timestamp_Label",
            src_col="Src IP Label",
            dst_col="Dst IP Label",
            label_col="Label_encoded",
            memory_dim=16,
            message_dim=64,
            hidden_dim=32,
            classifier_hidden_dim=32,
            lr=1e-3,
            epochs=200,
            save_dir=os.path.join(output_dir, "Darknet"),
        )
    else:
        raise ValueError(f"未知数据集: {dataset_name}")


def is_numeric(val):
    if pd.isna(val):
        return False
    try:
        float(val)
        return True
    except (ValueError, TypeError):
        return False


def has_chinese(text):
    if isinstance(text, str):
        return bool(re.search(r'[\u4e00-\u9fff]', text))
    return False


def check_row(row):
    for v in row:
        if not is_numeric(v):
            return False
        if has_chinese(v):
            return False
    return True


def _unify_raw_event_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    将原始 train_df / test_df / test_matrix.csv 统一成内部格式：
    - event_index
    - time
    - src_id
    - dst_id
    - label

    其余列保持原始边特征列不变。

    注意：
    这里强制按排序后的顺序重新生成 event_index，
    以保证和 memory_updates_each_event.csv 中按事件顺序记录的 event_index 一致。
    """
    df = df.copy()

    if "time" in df.columns:
        time_col = "time"
    elif "Timestamp_Label" in df.columns:
        time_col = "Timestamp_Label"
    else:
        raise ValueError("未找到时间列（time / Timestamp_Label）。")

    if "label" in df.columns:
        label_col = "label"
    elif "Label_encoded" in df.columns:
        label_col = "Label_encoded"
    else:
        raise ValueError("未找到标签列（label / Label_encoded）。")

    if "src_id" in df.columns:
        src_col = "src_id"
    else:
        raise ValueError(
            "未找到内部源节点列 src_id。当前脚本要求输入矩阵中必须包含映射后的 src_id。"
        )

    if "dst_id" in df.columns:
        dst_col = "dst_id"
    else:
        raise ValueError(
            "未找到内部目的节点列 dst_id。当前脚本要求输入矩阵中必须包含映射后的 dst_id。"
        )

    df = df.rename(columns={
        time_col: "time",
        label_col: "label",
        src_col: "src_id",
        dst_col: "dst_id",
    })

    df = sort_event_frame(df, "time")
    df["event_index"] = np.arange(len(df))

    return df

def _extract_event_level_gru_score(mem_df: pd.DataFrame) -> pd.DataFrame:
    """
    从 memory_updates_each_event.csv 中抽取事件级 GRU 异常分数。
    优先级：
        1) anomaly_score
        2) normal_prob -> 1 - normal_prob
        3) pred_prob   -> 1 - pred_prob
        4) 若都没有，则置 0
    返回：
        [event_index, gru_anomaly_score]
    """
    src_self_df = mem_df[mem_df["node_id"] == mem_df["src_id"]].copy()

    if "anomaly_score" in src_self_df.columns:
        score_df = src_self_df[["event_index", "anomaly_score"]].drop_duplicates(subset=["event_index"]).copy()
        score_df = score_df.rename(columns={"anomaly_score": "gru_anomaly_score"})
    elif "normal_prob" in src_self_df.columns:
        score_df = src_self_df[["event_index", "normal_prob"]].drop_duplicates(subset=["event_index"]).copy()
        score_df["gru_anomaly_score"] = 1.0 - score_df["normal_prob"].astype(float)
        score_df = score_df[["event_index", "gru_anomaly_score"]]
    elif "pred_prob" in src_self_df.columns:
        score_df = src_self_df[["event_index", "pred_prob"]].drop_duplicates(subset=["event_index"]).copy()
        score_df["gru_anomaly_score"] = 1.0 - score_df["pred_prob"].astype(float)
        score_df = score_df[["event_index", "gru_anomaly_score"]]
    else:
        score_df = src_self_df[["event_index"]].drop_duplicates(subset=["event_index"]).copy()
        score_df["gru_anomaly_score"] = 0.0

    return score_df

def load_memory_updates_as_event_df(
    file_path: str,
    raw_train_df: pd.DataFrame,
    raw_edge_feature_cols: List[str]
) -> pd.DataFrame:
    """
    将训练阶段保存的 memory_updates_each_event.csv（事件×节点快照表）
    转换为“每个事件一行”的训练事件级 DataFrame。

    输出统一为内部格式：
    - event_index
    - label
    - time
    - src_id
    - dst_id
    - gru_anomaly_score
    - src_mem_*
    - dst_mem_*
    - 原始边特征列 raw_edge_feature_cols
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"找不到训练文件: {file_path}")

    mem_df = pd.read_csv(file_path)

    required_cols = {"event_index", "time", "src_id", "dst_id", "label", "node_id"}
    missing = required_cols - set(mem_df.columns)
    if missing:
        raise ValueError(f"memory_updates_each_event.csv 缺少必要列: {missing}")

    mem_cols = [c for c in mem_df.columns if c.startswith("mem_")]
    if len(mem_cols) == 0:
        raise ValueError("未找到 mem_* 记忆列。")

    # 源节点自身记忆
    src_df = mem_df[mem_df["node_id"] == mem_df["src_id"]].copy()
    src_keep_cols = ["event_index", "time", "src_id", "dst_id", "label"] + mem_cols
    src_df = src_df[src_keep_cols].drop_duplicates(subset=["event_index"])
    src_df = src_df.rename(columns={c: f"src_{c}" for c in mem_cols})

    # 目的节点自身记忆
    dst_df = mem_df[mem_df["node_id"] == mem_df["dst_id"]].copy()
    dst_keep_cols = ["event_index"] + mem_cols
    dst_df = dst_df[dst_keep_cols].drop_duplicates(subset=["event_index"])
    dst_df = dst_df.rename(columns={c: f"dst_{c}" for c in mem_cols})

    # 合并成事件级表
    graph_df = pd.merge(src_df, dst_df, on="event_index", how="inner")

    graph_df = graph_df.rename(columns={
        "label": "label",
        "time": "time",
        "src_id": "src_id",
        "dst_id": "dst_id",
    })
    # merge 事件级 GRU 异常分数
    score_df = _extract_event_level_gru_score(mem_df)
    graph_df = pd.merge(
        graph_df,
        score_df,
        on="event_index",
        how="left"
    )

    if "gru_anomaly_score" not in graph_df.columns:
        graph_df["gru_anomaly_score"] = 0.0
    graph_df["gru_anomaly_score"] = graph_df["gru_anomaly_score"].fillna(0.0).astype(np.float32)

    # merge 原始训练边特征
    raw_train_df_std = _unify_raw_event_df_columns(raw_train_df)
    raw_edge_df = raw_train_df_std[
        ["event_index", "time", "src_id", "dst_id"] + raw_edge_feature_cols
    ].copy()

    graph_df = pd.merge(
        graph_df,
        raw_edge_df,
        on=["event_index", "time", "src_id", "dst_id"],
        how="left"
    )

    graph_df = graph_df.sort_values(
        by=["time", "event_index"]
    ).reset_index(drop=True)

    return graph_df


def load_trained_model(cfg, feature_dim: int, num_nodes: int):
    model_path = os.path.join(cfg.save_dir, "trained_GRU_memory_model.pt")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"找不到训练好的模型文件: {model_path}")

    model = NodeMemoryModel(
        num_nodes=num_nodes,
        feature_dim=feature_dim,
        memory_dim=cfg.memory_dim,
        message_dim=cfg.message_dim,
        hidden_dim=cfg.hidden_dim,
        classifier_hidden_dim=cfg.classifier_hidden_dim,
        device=cfg.device
    )

    state_dict = torch.load(model_path, map_location=cfg.device)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"已加载训练好的模型参数: {model_path}")
    return model


def load_final_memory_state(cfg, num_nodes: int, memory_dim: int):
    final_memory_path = os.path.join(cfg.save_dir, "final_memory_matrix.csv")
    if not os.path.isfile(final_memory_path):
        raise FileNotFoundError(f"找不到最终记忆矩阵文件: {final_memory_path}")

    final_memory_df = pd.read_csv(final_memory_path)

    required_cols = {"node_id", "last_update"}
    missing = required_cols - set(final_memory_df.columns)
    if missing:
        raise ValueError(f"final_memory_matrix.csv 缺少必要列: {missing}")

    mem_cols = [f"mem_{d}" for d in range(memory_dim)]
    missing_mem_cols = [c for c in mem_cols if c not in final_memory_df.columns]
    if missing_mem_cols:
        raise ValueError(f"final_memory_matrix.csv 缺少记忆列: {missing_mem_cols}")

    memory_matrix = torch.zeros(num_nodes, memory_dim, dtype=torch.float32, device=cfg.device)
    last_update = torch.zeros(num_nodes, dtype=torch.float32, device=cfg.device)

    for _, row in final_memory_df.iterrows():
        node_id = int(row["node_id"])
        if node_id < 0 or node_id >= num_nodes:
            raise ValueError(f"node_id={node_id} 越界，合法范围应为 [0, {num_nodes - 1}]")

        mem_vec = torch.tensor(
            row[mem_cols].values.astype(np.float32),
            dtype=torch.float32,
            device=cfg.device
        )
        memory_matrix[node_id] = mem_vec
        last_update[node_id] = float(row["last_update"])

    print(f"已加载最终记忆状态: {final_memory_path}")
    return memory_matrix, last_update


@torch.no_grad()
def load_test_matrix_as_event_df(
    file_path: str,
    cfg,
    raw_edge_feature_cols: List[str],
    num_nodes: int
) -> pd.DataFrame:
    """
    读取 test_matrix.csv，并从训练结束时的 final_memory_matrix.csv 继续更新，
    将测试集转换成和训练集一致的事件级格式。

    输出统一为内部格式：
    - event_index
    - label
    - time
    - src_id
    - dst_id
    - gru_anomaly_score
    - src_mem_*
    - dst_mem_*
    - 原始边特征列 raw_edge_feature_cols
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"找不到测试文件: {file_path}")

    test_df = pd.read_csv(file_path).copy()
    test_df = _unify_raw_event_df_columns(test_df)

    missing_raw_edge = [c for c in raw_edge_feature_cols if c not in test_df.columns]
    if missing_raw_edge:
        raise ValueError(f"test_matrix.csv 缺少原始边特征列: {missing_raw_edge}")

    model = load_trained_model(
        cfg=cfg,
        feature_dim=len(raw_edge_feature_cols),
        num_nodes=num_nodes
    )
    model.eval()

    memory_matrix, last_update = load_final_memory_state(
        cfg=cfg,
        num_nodes=num_nodes,
        memory_dim=cfg.memory_dim
    )

    records = []

    for _, row in test_df.iterrows():
        src_id = int(row["src_id"])
        dst_id = int(row["dst_id"])
        timestamp = float(row["time"])
        label = float(row["label"])

        event_feature = torch.tensor(
            row[raw_edge_feature_cols].values.astype(np.float32),
            dtype=torch.float32,
            device=cfg.device
        )

        src_new, dst_new, _, _, logit = model.compute_updated_states_and_score(
            src_id=src_id,
            dst_id=dst_id,
            timestamp=timestamp,
            event_feature=event_feature,
            memory_matrix=memory_matrix,
            last_update=last_update
        )
        normal_prob = torch.sigmoid(logit).item()
        gru_anomaly_score = 1.0 - normal_prob

        memory_matrix = memory_matrix.clone()
        memory_matrix[src_id] = src_new
        memory_matrix[dst_id] = dst_new

        last_update = last_update.clone()
        last_update[src_id] = timestamp
        last_update[dst_id] = timestamp

        record = {
            "event_index": int(row["event_index"]),
            "label": label,
            "time": timestamp,
            "src_id": src_id,
            "dst_id": dst_id,
            "gru_anomaly_score": float(gru_anomaly_score),
        }

        for d in range(cfg.memory_dim):
            record[f"src_mem_{d}"] = float(src_new[d].cpu().item())
        for d in range(cfg.memory_dim):
            record[f"dst_mem_{d}"] = float(dst_new[d].cpu().item())

        for c in raw_edge_feature_cols:
            record[c] = float(row[c])

        records.append(record)

    test_event_df = pd.DataFrame(records)
    return test_event_df


def align_feature_schema(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    检查训练集和测试集的事件级特征列是否一致；
    若顺序不同，则按训练集顺序重排测试集。
    """
    meta_cols = {
        "event_index",
        "label",
        "time",
        "src_id",
        "dst_id",
        "gru_anomaly_score"
    }

    train_feature_cols = [c for c in train_df.columns if c not in meta_cols]
    test_feature_cols = [c for c in test_df.columns if c not in meta_cols]

    train_feature_set = set(train_feature_cols)
    test_feature_set = set(test_feature_cols)

    missing_in_test = [c for c in train_feature_cols if c not in test_feature_set]
    extra_in_test = [c for c in test_feature_cols if c not in train_feature_set]

    if missing_in_test or extra_in_test:
        msg = ["训练集和测试集的特征列不一致。"]
        if missing_in_test:
            msg.append(f"测试集缺少特征列: {missing_in_test}")
        if extra_in_test:
            msg.append(f"测试集多出特征列: {extra_in_test}")
        raise ValueError("\n".join(msg))

    if train_feature_cols != test_feature_cols:
        print("训练集和测试集的特征列名称一致，但顺序不一致，已按训练集顺序重排测试集特征列。")
        ordered_cols = [
            "event_index",
            "label",
            "time",
            "src_id",
            "dst_id",
            "gru_anomaly_score"
        ] + train_feature_cols
        test_df = test_df[ordered_cols].copy()

    print("训练集和测试集的特征列已对齐。")
    return train_df, test_df, train_feature_cols

class TemporalBatchGraphSample:
    """
    按事件顺序构图，并支持反向 k-hop 历史边提取。

    训练模式:
        target_df = train_event_df
        base_history_df = None

    测试模式:
        target_df = test_event_df
        base_history_df = train_event_df

    对于 target_df 中每个事件：
        历史图 = base_history_df + 当前事件之前（含当前）的 target_df 历史

    额外约定：
    - 对重复有向边 (src_id, dst_id)，只保留最后一次出现
    - 保存目标事件边在 edgefeatures 中的位置 target_edge_index
    """

    def __init__(
        self,
        config,
        target_df: pd.DataFrame,
        base_history_df: pd.DataFrame = None,
        edge_feature_cols: List[str] = None
    ):
        self.config = config
        self.k_hop = getattr(config, "k_hop", 1)
        self.n_degree = getattr(config, "n_degree", None)

        self.src_col = "src_id"
        self.dst_col = "dst_id"
        self.time_col = "time"
        self.label_col = "label"
        self.event_id_col = "event_index"

        self.target_df = target_df.copy()
        self.base_history_df = None if base_history_df is None else base_history_df.copy()

        required_cols = {
            self.event_id_col,
            self.label_col,
            self.time_col,
            self.src_col,
            self.dst_col
        }

        missing_target = required_cols - set(self.target_df.columns)
        if missing_target:
            raise ValueError(f"target_df 缺少必要列: {missing_target}")

        if self.base_history_df is not None:
            missing_history = required_cols - set(self.base_history_df.columns)
            if missing_history:
                raise ValueError(f"base_history_df 缺少必要列: {missing_history}")

        # 节点特征列：只用节点自己的数值特征（src_mem_* / dst_mem_*）
        self.src_node_feature_cols = sorted(
            [c for c in self.target_df.columns if c.startswith("src_mem_")],
            key=lambda x: int(x.split("_")[-1])
        )
        self.dst_node_feature_cols = sorted(
            [c for c in self.target_df.columns if c.startswith("dst_mem_")],
            key=lambda x: int(x.split("_")[-1])
        )

        if len(self.src_node_feature_cols) == 0 or len(self.dst_node_feature_cols) == 0:
            raise ValueError("未找到 src_mem_* 或 dst_mem_* 节点特征列。")

        if len(self.src_node_feature_cols) != len(self.dst_node_feature_cols):
            raise ValueError("源节点特征维度与目的节点特征维度不一致。")

        self.edge_feature_cols = list(edge_feature_cols) if edge_feature_cols is not None else []

        if len(self.edge_feature_cols) > 0:
            missing_edge_target = [c for c in self.edge_feature_cols if c not in self.target_df.columns]
            if missing_edge_target:
                raise ValueError(f"target_df 缺少原始边特征列: {missing_edge_target}")

            if self.base_history_df is not None:
                missing_edge_history = [c for c in self.edge_feature_cols if c not in self.base_history_df.columns]
                if missing_edge_history:
                    raise ValueError(f"base_history_df 缺少原始边特征列: {missing_edge_history}")

        # 来源标记
        self.target_df["__source"] = "target"
        self.target_df["__source_rank"] = 1

        if self.base_history_df is not None:
            self.base_history_df["__source"] = "history"
            self.base_history_df["__source_rank"] = 0
            self.full_df = pd.concat(
                [self.base_history_df, self.target_df],
                ignore_index=True
            )
        else:
            self.full_df = self.target_df.copy()

        # 全局排序
        self.full_df = self.full_df.sort_values(
            by=[self.time_col, "__source_rank", self.event_id_col]
        ).reset_index(drop=True)

        self.full_df["__global_id"] = np.arange(len(self.full_df), dtype=np.int64)

        # =========================
        # 预缓存：减少后续 DataFrame 操作
        # =========================
        self.full_src = self.full_df[self.src_col].to_numpy(dtype=np.int64)
        self.full_dst = self.full_df[self.dst_col].to_numpy(dtype=np.int64)
         # 用于构造节点位置/身份信息：归一化后的节点 id
        self.max_node_id = int(
            getattr(self.config, "max_node_id", max(self.full_src.max(), self.full_dst.max()))
        )
        if self.max_node_id <= 0:
            self.max_node_id = 1

        self.full_time = self.full_df[self.time_col].to_numpy(dtype=np.float32)
        self.full_event_index = self.full_df[self.event_id_col].to_numpy(dtype=np.int64)
        self.full_source_rank = self.full_df["__source_rank"].to_numpy(dtype=np.int64)
        self.full_global_id = self.full_df["__global_id"].to_numpy(dtype=np.int64)
        self.full_source_flag = self.full_df["__source"].to_numpy()

        self.full_src_node_feat = self.full_df[self.src_node_feature_cols].to_numpy(dtype=np.float32)
        self.full_dst_node_feat = self.full_df[self.dst_node_feature_cols].to_numpy(dtype=np.float32)

        if len(self.edge_feature_cols) > 0:
            self.full_edge_feat = self.full_df[self.edge_feature_cols].to_numpy(dtype=np.float32)
        else:
            self.full_edge_feat = None

        # 目标事件在 full_df 中的位置（其实就等于 __global_id）
        self.target_global_ids = np.where(self.full_source_flag == "target")[0].astype(np.int64)

        # =========================
        # 全局反向边索引：只建一次
        # reverse_edge_rows[v] = [row_idx1, row_idx2, ...]
        # reverse_edge_srcs[v] = [src1, src2, ...]
        # 表示所有形如 src -> v 的历史边
        # =========================
        self.reverse_edge_rows = {}
        self.reverse_edge_srcs = {}

        for row_idx, (u, v) in enumerate(zip(self.full_src, self.full_dst)):
            if v not in self.reverse_edge_rows:
                self.reverse_edge_rows[v] = []
                self.reverse_edge_srcs[v] = []
            self.reverse_edge_rows[v].append(row_idx)
            self.reverse_edge_srcs[v].append(u)

    def normalize_sparse(self, mx):
        rowsum = np.array(mx.sum(1)).astype(np.float32)
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.0
        r_mat_inv = sp.diags(r_inv)
        return r_mat_inv.dot(mx)

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
        )
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)

    def _get_k_hop_edge_indices(self, global_id: int, target_src: int) -> set:
        """
        反向 k-hop 边提取（仅使用 row_idx <= global_id 的历史）：

        第1跳：找所有 u -> target_src
        第2跳：找所有 w -> u
        ...
        """
        frontier_dst_nodes = {target_src}
        visited_dst_nodes = {target_src}
        sampled_edge_indices = set()

        for _ in range(self.k_hop):
            next_frontier_dst_nodes = set()

            for dst_node in frontier_dst_nodes:
                if dst_node not in self.reverse_edge_rows:
                    continue

                rows = self.reverse_edge_rows[dst_node]
                srcs = self.reverse_edge_srcs[dst_node]

                # 只取历史中 row_idx <= global_id 的部分
                end_pos = bisect_right(rows, global_id)
                start_pos = 0
                if self.n_degree is not None and self.n_degree > 0:
                    start_pos = max(0, end_pos - self.n_degree)

                for pos in range(start_pos, end_pos):
                    row_idx = rows[pos]
                    src_node = srcs[pos]
                    sampled_edge_indices.add(row_idx)
                    next_frontier_dst_nodes.add(src_node)

            next_frontier_dst_nodes -= visited_dst_nodes
            visited_dst_nodes |= next_frontier_dst_nodes
            frontier_dst_nodes = next_frontier_dst_nodes

            if len(frontier_dst_nodes) == 0:
                break

        return sampled_edge_indices

    def _build_subgraph_edge_indices(
        self,
        global_id: int,
        src: int
    ) -> Tuple[List[int], int]:
        """
        构造当前目标事件对应的局部子图边索引列表。

        逻辑：
        1. 先按反向 k-hop 规则，从目标源节点 src 回溯历史边
        2. 再把当前目标事件本身加入
        3. 对重复有向边 (src_id, dst_id) 只保留最后一次出现
        4. 返回最终保留的全局行索引列表 kept_indices
        5. 返回目标事件边在 kept_indices 中的位置 target_edge_index
        """
        sampled_edge_indices = self._get_k_hop_edge_indices(global_id, src)

        # 当前目标事件本身加入
        selected_indices = set(sampled_edge_indices)
        selected_indices.add(global_id)

        # 按全局时序排序
        selected_sorted = sorted(selected_indices)

        # 对重复有向边，只保留最后一次出现
        seen_edges = set()
        kept_rev = []

        for idx in reversed(selected_sorted):
            edge_key = (int(self.full_src[idx]), int(self.full_dst[idx]))
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                kept_rev.append(idx)

        kept_indices = sorted(kept_rev)

        # 目标事件边一定要在最终结果里
        if global_id not in kept_indices:
            raise ValueError(f"目标事件边 global_id={global_id} 未保留在子图中。")

        target_edge_index = kept_indices.index(global_id)

        return kept_indices, target_edge_index

    def _build_node_mapping(self, kept_indices: List[int], src: int, dst: int):
        """
        固定:
            src -> 0
            dst -> 1
        其余节点按编号升序排列。
        """
        nodes = set()
        for idx in kept_indices:
            nodes.add(int(self.full_src[idx]))
            nodes.add(int(self.full_dst[idx]))

        nodes |= {src, dst}

        other_nodes = sorted([n for n in nodes if n not in {src, dst}])

        ordered_nodes = [src]
        if dst != src:
            ordered_nodes.append(dst)
        ordered_nodes.extend(other_nodes)

        node_mapping = {node: idx for idx, node in enumerate(ordered_nodes)}
        return ordered_nodes, node_mapping

    def _create_transition_matrix(self, edge_pairs, num_nodes):
        """
        构造节点-边转移矩阵 T: [N_v, N_e]
        """
        num_edges = len(edge_pairs)
        row_index = []
        col_index = []
        data = []

        for e_idx, (u, v) in enumerate(edge_pairs):
            row_index.extend([u, v])
            col_index.extend([e_idx, e_idx])
            data.extend([1.0, 1.0])

        T = sp.csr_matrix((data, (row_index, col_index)), shape=(num_nodes, num_edges))
        return T

    def _create_node_adj(self, edge_pairs, num_nodes):
        """
        构造节点邻接矩阵（有向）：
        若存在边 u -> v，则 adj[u, v] = 1
        """
        row_idx = []
        col_idx = []
        data = []

        for u, v in edge_pairs:
            row_idx.append(u)
            col_idx.append(v)
            data.append(1.0)

        if len(row_idx) == 0:
            adj = sp.eye(num_nodes, format="csr")
        else:
            adj = sp.csr_matrix((data, (row_idx, col_idx)), shape=(num_nodes, num_nodes))
            adj = adj + sp.eye(num_nodes, format="csr")

        return adj

    def _create_edge_adj(self, edge_pairs):
        """
        构造边邻接矩阵（有向）：
        若 e_i=(u_i,v_i), e_j=(u_j,v_j) 满足 v_i == u_j，且 i < j，
        则在 e_j 的聚合行连接历史边 e_i 的列，使当前事件表示可聚合历史消息。
        """
        num_edges = len(edge_pairs)
        if num_edges == 0:
            return sp.csr_matrix((0, 0), dtype=np.float32)

        row_idx = []
        col_idx = []
        data = []

        for i in range(num_edges):
            row_idx.append(i)
            col_idx.append(i)
            data.append(1.0)

        for i, (u_i, v_i) in enumerate(edge_pairs):
            for j, (u_j, v_j) in enumerate(edge_pairs):
                if i < j and v_i == u_j:
                    row_idx.append(j)
                    col_idx.append(i)
                    data.append(1.0)

        edge_adj = sp.csr_matrix((data, (row_idx, col_idx)), shape=(num_edges, num_edges))
        return edge_adj

    def _process_event_by_global_id(self, global_id: int):
        """
        对一个目标事件构造：
        - 节点特征矩阵
        - 边特征矩阵
        - 节点-边转移矩阵
        - 节点邻接矩阵
        - 边邻接矩阵
        - mask
        - target_edge_index: 目标事件边在 edgefeatures 中的位置
        """
        src = int(self.full_src[global_id])
        dst = int(self.full_dst[global_id])

        kept_indices, target_edge_index = self._build_subgraph_edge_indices(global_id, src)

        ordered_nodes, node_mapping = self._build_node_mapping(kept_indices, src, dst)
        num_nodes = len(ordered_nodes)

         # 节点特征矩阵：原节点记忆特征 + 1维归一化节点id
        base_node_feature_dim = len(self.src_node_feature_cols)
        node_feature_dim = base_node_feature_dim + 1
        node_features = np.zeros((num_nodes, node_feature_dim), dtype=np.float32)

        for idx in kept_indices:
            src_node = int(self.full_src[idx])
            dst_node = int(self.full_dst[idx])

            src_id_feat = np.array([src_node / self.max_node_id], dtype=np.float32)
            dst_id_feat = np.array([dst_node / self.max_node_id], dtype=np.float32)

            src_feat = np.concatenate([self.full_src_node_feat[idx], src_id_feat], axis=0)
            dst_feat = np.concatenate([self.full_dst_node_feat[idx], dst_id_feat], axis=0)

            node_features[node_mapping[src_node]] = src_feat
            node_features[node_mapping[dst_node]] = dst_feat

        mask = num_nodes

        edge_pairs = []
        for idx in kept_indices:
            u = node_mapping[int(self.full_src[idx])]
            v = node_mapping[int(self.full_dst[idx])]
            edge_pairs.append((u, v))

        if len(self.edge_feature_cols) == 0:
            edge_feature_matrix = np.zeros((len(kept_indices), 1), dtype=np.float32)
        else:
            edge_feature_matrix = self.full_edge_feat[kept_indices]

        if target_edge_index < 0 or target_edge_index >= len(edge_pairs):
            raise ValueError(
                f"target_edge_index 越界: {target_edge_index}, 边数={len(edge_pairs)}"
            )

        T = self._create_transition_matrix(edge_pairs, num_nodes)
        T = self.sparse_mx_to_torch_sparse_tensor(T)

        adj = self._create_node_adj(edge_pairs, num_nodes)
        adj = self.normalize_sparse(adj)
        adj = self.sparse_mx_to_torch_sparse_tensor(adj)

        eadj = self._create_edge_adj(edge_pairs)
        if eadj.shape[0] == 0:
            eadj = sp.csr_matrix([[1.0]], dtype=np.float32)
        else:
            eadj = self.normalize_sparse(eadj)
        eadj = self.sparse_mx_to_torch_sparse_tensor(eadj)

        return node_features, edge_feature_matrix, T, adj, eadj, mask, target_edge_index

    def get_batch_data(self):
        """
        只对 target_df 中的事件逐个构图并返回结果。
        额外返回 target_edge_indices，用于标记每个样本中目标事件边的位置。
        """
        results = []
        for global_id in tqdm(self.target_global_ids, total=len(self.target_global_ids), desc="Processing events"):
            results.append(self._process_event_by_global_id(int(global_id)))

        (
            all_node_features,
            all_edge_features,
            all_Tmat,
            all_adj,
            all_eadj,
            all_mask,
            all_target_edge_indices
        ) = zip(*results)

        return (
            list(all_node_features),
            list(all_edge_features),
            list(all_Tmat),
            list(all_adj),
            list(all_eadj),
            list(all_mask),
            list(all_target_edge_indices),
        )

if __name__ == "__main__":
    config = args
    if not hasattr(config, "k_hop"):
        config.k_hop = 1

    for config.data_set in ["IIoTAPT", "Darknet", "CloudDDoS"]:
        cfg = build_cfg_for_dataset(config.data_set, config.output_dir)
        cfg.k_hop = config.k_hop
        cfg.n_degree = config.n_degree

        dataset_dir = cfg.save_dir
        train_memory_file = os.path.join(dataset_dir, "memory_updates_each_event.csv")
        test_matrix_file = os.path.join(dataset_dir, "test_matrix.csv")

        if not os.path.isfile(train_memory_file):
            print(f"[跳过] {config.data_set} 未找到训练文件: {train_memory_file}")
            continue
        if not os.path.isfile(test_matrix_file):
            print(f"[跳过] {config.data_set} 未找到测试文件: {test_matrix_file}")
            continue

        print(f"\n====== 处理数据集: {config.data_set} ======")
        print(f"训练文件: {train_memory_file}")
        print(f"测试文件: {test_matrix_file}")
        print(f"k-hop: {cfg.k_hop}")

        # 原始训练/测试矩阵（已标准化，且包含 src_id/dst_id）
        train_df, test_df, node2id, id2node, raw_edge_feature_cols = load_and_prepare_data(cfg)
        num_nodes = len(node2id)

        # 训练事件级表：记忆特征 + 原始训练边特征
        train_event_df = load_memory_updates_as_event_df(
            file_path=train_memory_file,
            raw_train_df=train_df,
            raw_edge_feature_cols=raw_edge_feature_cols
        )

        # 测试事件级表：训练结束记忆继续更新 + 原始测试边特征
        test_event_df = load_test_matrix_as_event_df(
            file_path=test_matrix_file,
            cfg=cfg,
            raw_edge_feature_cols=raw_edge_feature_cols,
            num_nodes=num_nodes
        )

        # 对齐训练/测试事件级表的列模式
        train_event_df, test_event_df, _ = align_feature_schema(train_event_df, test_event_df)

        print(f"训练事件数: {len(train_event_df)}")
        print(f"测试事件数: {len(test_event_df)}")
        print(f"节点特征维度: {cfg.memory_dim + 1}")
        print(f"边特征维度: {len(raw_edge_feature_cols)}")

        # =========================
        # 训练图数据：仅训练历史
        # =========================
        train_label = train_event_df["label"].to_numpy()
        train_idx = train_event_df["event_index"].to_numpy()
        train_timestamp = train_event_df["time"].to_numpy()
        train_gru_scores = train_event_df["gru_anomaly_score"].to_numpy(dtype=np.float32)
        
        train_sampler = TemporalBatchGraphSample(
            config=cfg,
            target_df=train_event_df,
            base_history_df=None,
            edge_feature_cols=raw_edge_feature_cols
        )

        (
            train_node_features,
            train_edge_features,
            train_Tmat,
            train_adj,
            train_eadj,
            train_mask,
            train_target_edge_indices
        ) = train_sampler.get_batch_data()

        train_data = {
            "nodefeatures": train_node_features,
            "edgefeatures": train_edge_features,
            "masks": np.array(train_mask),
            "labels": train_label,
            "Tmats": train_Tmat,
            "adjs": train_adj,
            "eadjs": train_eadj,
            "event_index": train_idx,
            "time": train_timestamp,
            "target_edge_indices": np.array(train_target_edge_indices, dtype=np.int64),
            "gru_scores": train_gru_scores,
        }

        train_save_path = os.path.join(dataset_dir, f"{config.data_set}_train_khop_{cfg.k_hop}.pkl")
        with open(train_save_path, "wb") as f:
            pickle.dump(train_data, f)

        print(f"--Train Data OK-- 已保存到: {train_save_path}")

        # =========================
        # 测试图数据：训练历史 + 测试前缀历史
        # =========================
        test_label = test_event_df["label"].to_numpy()
        test_idx = test_event_df["event_index"].to_numpy()
        test_timestamp = test_event_df["time"].to_numpy()
        test_gru_scores = test_event_df["gru_anomaly_score"].to_numpy(dtype=np.float32)

        test_sampler = TemporalBatchGraphSample(
            config=cfg,
            target_df=test_event_df,
            base_history_df=train_event_df,
            edge_feature_cols=raw_edge_feature_cols
        )

        (
            test_node_features,
            test_edge_features,
            test_Tmat,
            test_adj,
            test_eadj,
            test_mask,
            test_target_edge_indices
        ) = test_sampler.get_batch_data()

        test_data = {
            "nodefeatures": test_node_features,
            "edgefeatures": test_edge_features,
            "masks": np.array(test_mask),
            "labels": test_label,
            "Tmats": test_Tmat,
            "adjs": test_adj,
            "eadjs": test_eadj,
            "event_index": test_idx,
            "time": test_timestamp,
            "target_edge_indices": np.array(test_target_edge_indices, dtype=np.int64),
            "gru_scores": test_gru_scores,
        }

        test_save_path = os.path.join(dataset_dir, f"{config.data_set}_test_khop_{cfg.k_hop}.pkl")
        with open(test_save_path, "wb") as f:
            pickle.dump(test_data, f)

        print(f"--Test Data OK-- 已保存到: {test_save_path}")
