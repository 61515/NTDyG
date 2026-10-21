import os
import json
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


# =========================
# 1. 配置区,数据调整
# =========================
@dataclass
# class Config:
#     csv_path: str = "dataset/IIoTAPT_re.csv"          # csv文件路径
#     train_idx: str = "dataset/IIoTAPT_finetune_indx.csv"          # csv文件路径
#     test_idx: str = "dataset/IIoTAPT_test_indx.csv"          # csv文件路径

#     time_col: str = "Timestamp_Label"              # 时间列名
#     src_col: str = "Src IP Label"                # 源节点列名
#     dst_col: str = "Dst IP Label"                # 目的节点列名
#     label_col: str = "Label_encoded"            # 标签列名（0/1）
#     feature_cols: List[str] = None      # 特征列名列表；None 表示自动推断

#     memory_dim: int = 16                # 节点记忆维度 d_m
#     message_dim: int = 52               # 消息维度
#     hidden_dim: int = 32                # 消息 MLP 隐层维度
#     classifier_hidden_dim: int = 32     # 分类器隐层维度

#     lr: float = 1e-3
#     epochs: int = 10
#     weight_decay: float = 1e-5
#     device: str = "cuda:{}".format(0) if torch.cuda.is_available() else "cpu"

#     save_dir: str = "outputs/IIoTAPT"
#     record_every_event: bool = True     # True: 每条事件后记录记忆矩阵
#     seed: int = 42
#     num_negatives: int = 3

# class Config:
#     csv_path: str = "dataset/CloudDDoS_re.csv"          # csv文件路径
#     train_idx: str = "dataset/CloudDDoS_finetune_indx.csv"          # csv文件路径
#     test_idx: str = "dataset/CloudDDoS_test_indx.csv"          # csv文件路径

#     time_col: str = "Timestamp_Label"              # 时间列名
#     src_col: str = "Src IP Label"                # 源节点列名
#     dst_col: str = "Dst IP Label"                # 目的节点列名
#     label_col: str = "Label_encoded"            # 标签列名（0/1）
#     feature_cols: List[str] = None      # 特征列名列表；None 表示自动推断

#     memory_dim: int = 16                # 节点记忆维度 d_m
#     message_dim: int = 314               # 消息维度
#     hidden_dim: int = 32                # 消息 MLP 隐层维度
#     classifier_hidden_dim: int = 32     # 分类器隐层维度

#     lr: float = 1e-3
#     epochs: int = 10
#     weight_decay: float = 1e-5
#     device: str = "cuda:{}".format(0) if torch.cuda.is_available() else "cpu"

#     save_dir: str = "outputs/CloudDDoS"
#     record_every_event: bool = True     # True: 每条事件后记录记忆矩阵
#     seed: int = 42
#     num_negatives: int = 3
class Config:
    csv_path: str = "dataset/Darknet_re.csv"          # csv文件路径
    train_idx: str = "dataset/Darknet_finetune_indx.csv"          # csv文件路径
    test_idx: str = "dataset/Darknet_test_indx.csv"          # csv文件路径

    time_col: str = "Timestamp_Label"              # 时间列名
    src_col: str = "Src IP Label"                # 源节点列名
    dst_col: str = "Dst IP Label"                # 目的节点列名
    label_col: str = "Label_encoded"            # 标签列名（0/1）
    feature_cols: List[str] = None      # 特征列名列表；None 表示自动推断

    memory_dim: int = 16                # 节点记忆维度 d_m
    message_dim: int = 64               # 消息维度
    hidden_dim: int = 32                # 消息 MLP 隐层维度
    classifier_hidden_dim: int = 32     # 分类器隐层维度

    lr: float = 1e-3
    epochs: int = 10
    weight_decay: float = 1e-5
    device: str = "cuda:{}".format(0) if torch.cuda.is_available() else "cpu"

    save_dir: str = "outputs/Darknet"
    record_every_event: bool = True     # True: 每条事件后记录记忆矩阵
    seed: int = 42
    num_negatives: int = 3
# =========================
# 2. 工具函数
# =========================
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

# 标准化处理
def standardize_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    means = df[feature_cols].mean()
    stds = df[feature_cols].std().replace(0, 1.0)
    df[feature_cols] = (df[feature_cols] - means) / stds
    return df

def fit_standardizer(df: pd.DataFrame, feature_cols: List[str]):
    means = df[feature_cols].mean()
    stds = df[feature_cols].std().replace(0, 1.0)
    return means, stds

def apply_standardizer(df: pd.DataFrame, feature_cols: List[str], means, stds) -> pd.DataFrame:
    df = df.copy()
    df[feature_cols] = (df[feature_cols] - means) / stds
    return df

# 自动推断哪些列是特征列
def infer_feature_cols(df: pd.DataFrame, cfg: Config) -> List[str]:
    excluded = {cfg.time_col, cfg.src_col, cfg.dst_col, cfg.label_col}
    cols = [c for c in df.columns if c not in excluded]
    if not cols:
        raise ValueError("未找到特征列，请在 Config.feature_cols 中显式指定。")
    return cols


# =========================
# 3. 数据读取与预处理
# =========================
def load_index_file(index_path: str) -> List[int]:
    """
    读取无标题行的索引文件。
    默认每行一个整数索引。
    """
    idx_df = pd.read_csv(index_path, header=None)
    indices = idx_df.iloc[:, 0].astype(int).tolist()
    return indices

def load_and_prepare_data(cfg: Config)  -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, int], Dict[int, int], List[str]]:
    """
    返回:
        train_df: 处理后的训练集
        test_df: 处理后的测试集
        node2id: 原始整数节点 -> 连续索引
        id2node: 连续索引 -> 原始整数节点
        feature_cols: 特征列名列表
    """
    df = pd.read_csv(cfg.csv_path)

    required_cols = {cfg.time_col, cfg.src_col, cfg.dst_col, cfg.label_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV 缺少必要列: {missing}")

    if cfg.feature_cols is None:
        feature_cols = infer_feature_cols(df, cfg)
    else:
        feature_cols = cfg.feature_cols
        missing_feat = set(feature_cols) - set(df.columns)
        if missing_feat:
            raise ValueError(f"CSV 缺少指定特征列: {missing_feat}")
    
    # 4) 建立全局节点映射
    #    注意：即使 src/dst 已经是整数，也仍建议映射成从 0 开始的连续索引，
    #    否则后续若直接作为张量索引，可能越界或浪费大量内存。
    unique_nodes = pd.Index(
        df[cfg.src_col].astype(int).tolist() + df[cfg.dst_col].astype(int).tolist()
    ).unique()
    node2id = {node: idx for idx, node in enumerate(unique_nodes)}
    id2node = {idx: node for node, idx in node2id.items()}

    train_indices = load_index_file(cfg.train_idx)
    test_indices = load_index_file(cfg.test_idx)
    max_idx = len(df) - 1
    if len(train_indices) == 0:
        raise ValueError("训练索引文件为空。")
    if len(test_indices) == 0:
        raise ValueError("测试索引文件为空。")
    if min(train_indices) < 0 or max(train_indices) > max_idx:
        raise ValueError(f"训练索引越界，合法范围应为 [0, {max_idx}]")
    if min(test_indices) < 0 or max(test_indices) > max_idx:
        raise ValueError(f"测试索引越界，合法范围应为 [0, {max_idx}]")
    
    train_df = df.iloc[train_indices].copy().reset_index(drop=True)
    test_df = df.iloc[test_indices].copy().reset_index(drop=True)

    # 7) 使用全局 node2id 生成连续节点编号
    train_df["src_id"] = train_df[cfg.src_col].astype(int).map(node2id)
    train_df["dst_id"] = train_df[cfg.dst_col].astype(int).map(node2id)

    test_df["src_id"] = test_df[cfg.src_col].astype(int).map(node2id)
    test_df["dst_id"] = test_df[cfg.dst_col].astype(int).map(node2id)

    train_df[cfg.label_col] = train_df[cfg.label_col].astype(float)
    test_df[cfg.label_col] = test_df[cfg.label_col].astype(float)


    means, stds = fit_standardizer(train_df, feature_cols)
    train_df = apply_standardizer(train_df, feature_cols, means, stds)
    test_df = apply_standardizer(test_df, feature_cols, means, stds)

    train_df = train_df.sort_values(by=[cfg.time_col]).reset_index(drop=True)
    test_df = test_df.sort_values(by=[cfg.time_col]).reset_index(drop=True)

    

    return train_df, test_df, node2id, id2node, feature_cols

# =========================
# 4. 模型定义
# =========================
class NodeMemoryModel(nn.Module):
    """
    一个共享 GRUCell 的节点记忆模型：
    - 全局维护 memory_matrix: [num_nodes, memory_dim]
    - 每条事件构造 src/dst 两个消息
    - 用同一个 GRUCell 分别更新 src 和 dst 的记忆
    - 用更新后的 src/dst 记忆 + 当前事件特征 做正常性打分
    """
    def __init__(self, num_nodes: int, feature_dim: int, memory_dim: int, message_dim: int,
                 hidden_dim: int, classifier_hidden_dim: int, device: str):
        super().__init__()
        self.num_nodes = num_nodes
        self.feature_dim = feature_dim
        self.memory_dim = memory_dim
        self.message_dim = message_dim
        self.device = device

        # 消息构造器
        # 输入: [m_src, m_dst, x_e, delta_t] -> message
        # 维度 = memory_dim + memory_dim + feature_dim + 1
        msg_input_dim = 2 * memory_dim + feature_dim + 1
        self.message_mlp = nn.Sequential(
            nn.Linear(msg_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, message_dim)
        )

        # 共享的 GRUCell
        self.memory_updater = nn.GRUCell(input_size=message_dim, hidden_size=memory_dim)

        # 正常性打分头
        # 输入: [m_src, m_dst, x_e]
        # 输出越大表示越像正常事件
        # 输入: [m_src_new, m_dst_new, x_e]
        cls_input_dim = 2 * memory_dim + feature_dim
        self.classifier = nn.Sequential(
            nn.Linear(cls_input_dim, classifier_hidden_dim),
            nn.ReLU(),
            nn.Linear(classifier_hidden_dim, 1)
        )

        self.to(device)

    def init_memory(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        初始化: 初始化每个节点的记忆状态
        - memory_matrix: [num_nodes, memory_dim]
        - last_update: [num_nodes]
        """
        # 节点记忆矩阵 [num_nodes,memory_dim]
        memory_matrix = torch.zeros(self.num_nodes, self.memory_dim, device=self.device)
        # 每个节点最近一次被更新的时间，形状是：[num_nodes]
        last_update = torch.zeros(self.num_nodes, device=self.device)
        return memory_matrix, last_update

    def build_message(
        self,
        src_memory: torch.Tensor,
        dst_memory: torch.Tensor,
        event_feature: torch.Tensor,
        delta_t: torch.Tensor
    ) -> torch.Tensor:
        """
        输入: 根据一次事件交互构造消息向量，供后续 GRU 更新记忆使用
            src_memory: [memory_dim]
            dst_memory: [memory_dim]
            event_feature: [feature_dim]
            delta_t: [1]
        输出:
            message: [message_dim]
        """
        # 针对一条事件构造消息向量
        x = torch.cat([src_memory, dst_memory, event_feature, delta_t], dim=0)
        return self.message_mlp(x)
    
    def score_event(
        self,
        src_memory: torch.Tensor,
        dst_memory: torch.Tensor,
        event_feature: torch.Tensor
    ) -> torch.Tensor:
        """
        对事件做正常性打分。
        输出:
            logit: 标量，越大表示越像正常事件
        """
        cls_input = torch.cat([src_memory, dst_memory, event_feature], dim=0)
        logit = self.classifier(cls_input).squeeze(0)
        return logit
    
    def compute_updated_states_and_score(
        self,
        src_id: int,
        dst_id: int,
        timestamp: float,
        event_feature: torch.Tensor,
        memory_matrix: torch.Tensor,
        last_update: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        基于当前历史状态，计算该事件发生后的:
        - src_new
        - dst_new
        - src_delta_t
        - dst_delta_t
        - logit (基于更新后的 src_new / dst_new 打分)

        注意:
        这里只做“当前事件对应的一次局部更新推演”，不负责写回全局 memory_matrix。
        """
        src_old = memory_matrix[src_id]
        dst_old = memory_matrix[dst_id]

        src_delta_t = torch.tensor(
            [timestamp], dtype=torch.float32, device=self.device
        ) - last_update[src_id].unsqueeze(0)

        dst_delta_t = torch.tensor(
            [timestamp], dtype=torch.float32, device=self.device
        ) - last_update[dst_id].unsqueeze(0)

        src_msg = self.build_message(src_old, dst_old, event_feature, src_delta_t)
        dst_msg = self.build_message(dst_old, src_old, event_feature, dst_delta_t)

        src_new = self.memory_updater(
            src_msg.unsqueeze(0), src_old.unsqueeze(0)
        ).squeeze(0)

        dst_new = self.memory_updater(
            dst_msg.unsqueeze(0), dst_old.unsqueeze(0)
        ).squeeze(0)

        # 用更新后的状态打分
        logit = self.score_event(src_new, dst_new, event_feature)

        return src_new, dst_new, src_delta_t, dst_delta_t, logit
    
    def update_one_event(
        self,
        src_id: int,
        dst_id: int,
        timestamp: float,
        event_feature: torch.Tensor,
        memory_matrix: torch.Tensor,
        last_update: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        对单条事件执行记忆更新，并输出当前事件预测 logit

        输入:
            src_id, dst_id: 节点编号
            timestamp: 当前事件时间
            event_feature: [feature_dim]
            memory_matrix: [num_nodes, memory_dim]
            last_update: [num_nodes]

        输出:
            new_memory_matrix
            new_last_update
            pos_logit: 当前真实事件的正常性得分（更新前计算）
        """
        # src_old = memory_matrix[src_id]
        # dst_old = memory_matrix[dst_id]
        # # 用更新前状态对真实事件打分
        # pos_logit = self.score_event(src_old, dst_old, event_feature)

        # src_delta_t = torch.tensor([timestamp], device=self.device) - last_update[src_id].unsqueeze(0)
        # dst_delta_t = torch.tensor([timestamp], device=self.device) - last_update[dst_id].unsqueeze(0)

        # # 为 src / dst 分别构造消息
        # src_msg = self.build_message(src_old, dst_old, event_feature, src_delta_t)
        # dst_msg = self.build_message(dst_old, src_old, event_feature, dst_delta_t)

        # # 用共享 GRUCell 更新
        # src_new = self.memory_updater(src_msg.unsqueeze(0), src_old.unsqueeze(0)).squeeze(0)
        # dst_new = self.memory_updater(dst_msg.unsqueeze(0), dst_old.unsqueeze(0)).squeeze(0)
        src_new, dst_new, _, _, logit = self.compute_updated_states_and_score(
            src_id=src_id,
            dst_id=dst_id,
            timestamp=timestamp,
            event_feature=event_feature,
            memory_matrix=memory_matrix,
            last_update=last_update
        )

        # 写回新的记忆矩阵（克隆后写，避免原地操作破坏 autograd）
        new_memory = memory_matrix.clone()
        new_memory[src_id] = src_new
        new_memory[dst_id] = dst_new

        # 更新时间
        new_last_update = last_update.clone()
        new_last_update[src_id] = timestamp
        new_last_update[dst_id] = timestamp

        # 用更新后的记忆做当前事件分类
        # cls_input = torch.cat([src_new, dst_new, event_feature], dim=0)
        # logit = self.classifier(cls_input).squeeze(0)

        return new_memory, new_last_update, logit
    
def sample_negative_dst(num_nodes: int, src_id: int, true_dst_id: int) -> int:
    """
    随机采样一个负目的节点
    约束:
    - 不能等于真实 dst
    - 不能等于 src
    """
    while True:
        neg_dst_id = np.random.randint(0, num_nodes)
        if neg_dst_id != true_dst_id and neg_dst_id != src_id:
            return neg_dst_id


def bpr_loss(pos_logit: torch.Tensor, neg_logit: torch.Tensor) -> torch.Tensor:
    """
    BPR loss:
        -log sigmoid(pos_logit - neg_logit)
    """
    return -torch.log(torch.sigmoid(pos_logit - neg_logit) + 1e-12)

# =========================
# 5. 训练
# =========================
def train_model(cfg: Config, df: pd.DataFrame, feature_cols: List[str], num_nodes: int) -> NodeMemoryModel:
    device = cfg.device
    feature_dim = len(feature_cols)
    # 只用正常事件训练
    normal_df = df[df[cfg.label_col] == 0].copy().reset_index(drop=True)
    if len(normal_df) == 0:
        raise ValueError("训练集中没有正常事件label==0, 无法进行BPR训练。")
    
    model = NodeMemoryModel(
        num_nodes=num_nodes,
        feature_dim=feature_dim,
        memory_dim=cfg.memory_dim,
        message_dim=cfg.message_dim,
        hidden_dim=cfg.hidden_dim,
        classifier_hidden_dim=cfg.classifier_hidden_dim,
        device=device
    )

    # criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    print(f"开始训练: device={device}, num_nodes={num_nodes}, feature_dim={feature_dim}")
    print(f"BPR训练使用的正常事件数: {len(normal_df)}")

    for epoch in range(cfg.epochs):
        model.train()

        # 每个 epoch 从零初始化记忆
        memory_matrix, last_update = model.init_memory()

        total_loss = 0.0
        total_events = 0

        optimizer.zero_grad()

        # 逐条事件前向
        for _, row in df.iterrows():
            src_id = int(row["src_id"])
            dst_id = int(row["dst_id"])
            timestamp = float(row[cfg.time_col])
            # label = torch.tensor(float(row[cfg.label_col]), dtype=torch.float32, device=device)

            event_feature = torch.tensor(
                row[feature_cols].values.astype(np.float32),
                dtype=torch.float32,
                device=device
            )
            
            # ---------- 正样本：真实正常事件 ----------
            # 基于“更新后的状态”打分
            pos_src_new, pos_dst_new, _, _, pos_logit = model.compute_updated_states_and_score(
                src_id=src_id,
                dst_id=dst_id,
                timestamp=timestamp,
                event_feature=event_feature,
                memory_matrix=memory_matrix,
                last_update=last_update
            )
            # ---------- 负样本：随机替换目的节点 ----------
            neg_losses = []
            for _ in range(cfg.num_negatives):
                neg_dst_id = sample_negative_dst(
                    num_nodes=num_nodes,
                    src_id=src_id,
                    true_dst_id=dst_id
                )

                # 负样本同样基于“同一个更新前历史状态”做一次对应更新，再打分
                _, _, _, _, neg_logit = model.compute_updated_states_and_score(
                    src_id=src_id,
                    dst_id=neg_dst_id,
                    timestamp=timestamp,
                    event_feature=event_feature,
                    memory_matrix=memory_matrix,
                    last_update=last_update
                )

                neg_losses.append(bpr_loss(pos_logit, neg_logit))

            # memory_matrix, last_update, logit = model.update_one_event(
            #     src_id=src_id,
            #     dst_id=dst_id,
            #     timestamp=timestamp,
            #     event_feature=event_feature,
            #     memory_matrix=memory_matrix,
            #     last_update=last_update
            # )

            # loss = criterion(logit.view(1), label.view(1))
            # total_loss = total_loss + loss
            # total_events += 1
            loss = torch.stack(neg_losses).mean()

            total_loss = total_loss + loss
            total_events += 1

            # ---------- 只把真实事件写回全局记忆 ----------
            memory_matrix = memory_matrix.clone()
            memory_matrix[src_id] = pos_src_new
            memory_matrix[dst_id] = pos_dst_new

            last_update = last_update.clone()
            last_update[src_id] = timestamp
            last_update[dst_id] = timestamp

        avg_loss = total_loss / max(total_events, 1)
        avg_loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch + 1}/{cfg.epochs}] - avg_loss: {avg_loss.item():.6f}")

    return model
    #     avg_loss = total_loss / max(total_events, 1)
    #     avg_loss.backward()
    #     optimizer.step()

    #     print(f"Epoch [{epoch + 1}/{cfg.epochs}] - avg_loss: {avg_loss.item():.6f}")

    # return model


# =========================
# 6. 用训练好的模型做一遍最终记忆更新，并记录全过程
# =========================
@torch.no_grad()
def run_and_record_memory_updates(
    cfg: Config,
    model: NodeMemoryModel,
    df: pd.DataFrame,
    feature_cols: List[str],
    id2node: Dict[int, str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    device = cfg.device
    model.eval()

    memory_matrix, last_update = model.init_memory()

    memory_update_records = []
    num_nodes = memory_matrix.shape[0]
    memory_dim = memory_matrix.shape[1]

    for event_idx, row in df.iterrows():
        src_id = int(row["src_id"])
        dst_id = int(row["dst_id"])
        timestamp = float(row[cfg.time_col])
        label = float(row[cfg.label_col])

        event_feature = torch.tensor(
            row[feature_cols].values.astype(np.float32),
            dtype=torch.float32,
            device=device
        )

        # memory_matrix, last_update, logit = model.update_one_event(
        #     src_id=src_id,
        #     dst_id=dst_id,
        #     timestamp=timestamp,
        #     event_feature=event_feature,
        #     memory_matrix=memory_matrix,
        #     last_update=last_update
        # )
         # 基于更新后的状态计算真实事件分数
        src_new, dst_new, _, _, logit = model.compute_updated_states_and_score(
            src_id=src_id,
            dst_id=dst_id,
            timestamp=timestamp,
            event_feature=event_feature,
            memory_matrix=memory_matrix,
            last_update=last_update
        )

        # pred_prob = torch.sigmoid(logit).item()

        # # 记录当前事件后的完整记忆矩阵
        # current_memory_np = memory_matrix.cpu().numpy()
        # 写回真实事件更新
        memory_matrix = memory_matrix.clone()
        memory_matrix[src_id] = src_new
        memory_matrix[dst_id] = dst_new

        last_update = last_update.clone()
        last_update[src_id] = timestamp
        last_update[dst_id] = timestamp

        normal_prob = torch.sigmoid(logit).item()
        anomaly_score = 1.0 - normal_prob

        current_memory_np = memory_matrix.cpu().numpy()

        for node_id in range(num_nodes):
            record = {
                "event_index": event_idx,
                "time": timestamp,
                "src_id": src_id,
                "dst_id": dst_id,
                "src_node": id2node[src_id],
                "dst_node": id2node[dst_id],
                "label": label,
                "normal_prob": normal_prob,
                "anomaly_score": anomaly_score,
                "node_id": node_id, # 当前正在保存记忆的节点编号
                "node_name": id2node[node_id], # 当前节点的原始名称
            }
            for d in range(memory_dim):
                record[f"mem_{d}"] = current_memory_np[node_id, d]
            #把节点 node_id 当前记忆向量的每一维都拆开存到字典里。
            memory_update_records.append(record) # 每个事件之后、每个节点的记忆快照

    # 最终记忆矩阵 所有事件全部处理结束后，每个节点最终的记忆向量和最后更新时间。
    final_memory_np = memory_matrix.cpu().numpy()
    final_records = []
    for node_id in range(num_nodes):
        record = {
            "node_id": node_id,
            "node_name": id2node[node_id],
            "last_update": float(last_update[node_id].cpu().item())
        }
        for d in range(memory_dim):
            record[f"mem_{d}"] = final_memory_np[node_id, d]
        final_records.append(record)

    memory_update_df = pd.DataFrame(memory_update_records)
    final_memory_df = pd.DataFrame(final_records) # 所有事件处理结束后的最终记忆状态

    return memory_update_df, final_memory_df


# =========================
# 7. 保存最终 GRU 参数
# =========================
def save_gru_parameters(cfg: Config, model: NodeMemoryModel) -> None:
    """
    保存整个 NodeMemoryModel 的所有可学习参数，而不仅仅是 GRU 参数。
    同时保存为:
    1) JSON: 便于人工查看
    2) PT:   便于后续直接加载
    """

    model_state = model.state_dict()

    save_json = {}
    for k, v in model_state.items():
        save_json[k] = v.detach().cpu().numpy().tolist()

    json_path = os.path.join(cfg.save_dir, "final_model_parameters.json")
    pt_path = os.path.join(cfg.save_dir, "final_model_parameters.pt")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(save_json, f, ensure_ascii=False, indent=2)

    torch.save(model_state, pt_path)

    print(f"已保存 GRU 参数到: {json_path}")
    print(f"已保存 GRU 参数到: {pt_path}")


# =========================
# 8. 主函数
# =========================
def main():
    cfg = Config(
        # csv_path="dataset/CloudDDoS_re.csv",          # csv文件路径
        # csv_path="dataset/IIoTAPT_re.csv",      # 改成你的文件路径
        csv_path = "dataset/Darknet_re.csv",          # csv文件路径
        time_col="Timestamp_Label",          # 改成你的时间列名
        src_col="Src IP Label",            # 改成你的源节点列名
        dst_col="Dst IP Label",            # 改成你的目的节点列名
        label_col="Label_encoded",        # 改成你的标签列名
        feature_cols=None,        # 若为 None，则自动把其余列都当作特征
        memory_dim=16,
        # message_dim=52,
        # message_dim=314,
        message_dim=64,
        hidden_dim=32,
        classifier_hidden_dim=32,
        lr=1e-3,
        epochs=200
    )


    set_seed(cfg.seed)
    ensure_dir(cfg.save_dir)

    # df, node2id, id2node, feature_cols = load_and_prepare_data(cfg)
    train_df, test_df, node2id, id2node, feature_cols = load_and_prepare_data(cfg)
    train_path = os.path.join(cfg.save_dir, "train_matrix.csv")
    train_df.to_csv(train_path, index=False, encoding="utf-8-sig")

    test_path = os.path.join(cfg.save_dir, "test_matrix.csv")
    test_df.to_csv(test_path, index=False, encoding="utf-8-sig")
    print(f"已保存test")

    num_nodes = len(node2id)

    print("数据读取完成")
    print(f"训练样本数: {len(train_df)}")
    train_nodes = pd.Index(train_df["src_id"].tolist() + train_df["dst_id"].tolist()).unique()
    train_num_nodes = len(train_nodes)
    print(f"节点数: {train_num_nodes}")
    print(f"特征列: {feature_cols}")

    model = train_model(cfg, train_df, feature_cols, num_nodes)

    # 最终用训练好的模型重新跑一遍，并记录每个时刻记忆矩阵
    memory_update_df, final_memory_df = run_and_record_memory_updates(
        cfg=cfg,
        model=model,
        df=train_df,
        feature_cols=feature_cols,
        id2node=id2node
    )

    updates_path = os.path.join(cfg.save_dir, "memory_updates_each_event.csv")
    final_memory_path = os.path.join(cfg.save_dir, "final_memory_matrix.csv")
    model_path = os.path.join(cfg.save_dir, "trained_GRU_memory_model.pt")

    memory_update_df.to_csv(updates_path, index=False, encoding="utf-8-sig")
    final_memory_df.to_csv(final_memory_path, index=False, encoding="utf-8-sig")
    torch.save(model.state_dict(), model_path)

    # save_gru_parameters(cfg, model)

    print(f"已保存每个时刻的记忆矩阵更新到: {updates_path}")
    print(f"已保存最终记忆矩阵到: {final_memory_path}")
    print(f"已保存整个模型参数到: {model_path}")


if __name__ == "__main__":
    main()