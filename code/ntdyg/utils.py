import numpy as np
import random, logging
import datetime, os
import torch
from torch import nn
import numpy as np
import random
import os
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize

def setup_logger(log_path):
    """
    创建一个 logger，输出到屏幕和指定 log_path 文件。
    如果 log_path 所在目录不存在，会自动创建。
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger(log_path)  # 用路径作为名字，便于区分不同数据集
    logger.setLevel(logging.DEBUG)
    # 如果已经添加过 handler，先移除，避免重复日志
    if logger.hasHandlers():
        logger.handlers.clear()
    # 控制台 handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    # 文件 handler
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

class EarlyStopMonitor(object):
    def __init__(self, max_round=10, higher_better=True, tolerance=1e-3):
        self.max_round = max_round # 在“无改善”后最多允许多少轮（epoch）连续停滞
        self.num_round = 0  # 连续未改善的轮次数

        self.epoch_count = 0  # 已经检查过的总 epoch 数
        self.best_epoch = 0     # 最近一次更新最佳模型的 epoch

        self.last_best = None # 记录“最好”指标的数值（可能正向或反向）
        self.higher_better = higher_better # 如果目标是“越大越好”（如准确率），则 True；否则（如损失）设 False
        self.tolerance = tolerance # 相对改善阈值：只有超过 tol 才算“真正”提升

    def early_stop_check(self, curr_val):
        """
        每过一轮训练／验证后调用一次，把当下指标 curr_val 传进来：
          - 如果返回 True，则说明已经连续 max_round 轮没有显著改善，应当停止训练。
          - 如果返回 False，则训练可继续。
        """
        self.epoch_count += 1
        # 如果是“越小越好”（如损失），则把指标取负，让后续统一用“越大越好”的判断
        if not self.higher_better:
            curr_val *= -1
        # 第一次调用时，直接把 last_best 设为当前值
        if self.last_best is None:
            self.last_best = curr_val
        # 计算相对提升：(curr_val - last_best) / |last_best|
        elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
            # 有“真正”提升：更新 last_best，重置计数，记录最佳 epoch
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
        else:
            # 未达阈值视为无改善，连续计数 +1
            self.num_round += 1
        # 如果连续“无改善”轮数达到 max_round，就返回 True，触发早停
        return self.num_round >= self.max_round


def set_seed(seed): # 尽可能地固定随机性
        random.seed(seed) # 固定 Python 标准库 random 模块的随机数生成器种子。
        np.random.seed(seed) # 固定 numpy 模块的随机数生成器种子。
        torch.manual_seed(seed) # 设置 CPU 上所有 PyTorch 随机数生成器的种子。
        torch.cuda.manual_seed(seed) #  设置 GPU 上所有 PyTorch 随机数生成器的种子。
        torch.cuda.manual_seed_all(seed) # 设置所有 GPU 上所有 PyTorch 随机数生成器的种子。
        torch.use_deterministic_algorithms(True) # 强制 PyTorch 在支持的操作（ops）上使用确定性算法。部分操作在不同硬件／库版本上会有非确定性实现，开启此项后如果调用了不支持确定性的 op，将抛错提醒你。
        torch.backends.cudnn.deterministic = True # 让 cuDNN（NVIDIA GPU 加速库）只使用确定性算法（如在卷积操作上）。
        torch.backends.cudnn.benchmark = False # 禁用 cuDNN 的自动调优功能，防止它为了速度在运行时挑选最优算法而引入随机性。
        torch.backends.cudnn.enabled = False # 禁用 cuDNN 的自动寻找最优算法，防止它为运行时选择最优算法引入随机性。
        os.environ['PYTHONHASHSEED'] = str(seed) # 设置环境变量 PYTHONHASHSEED，让 Python 的哈希函数也表现确定（影响字典遍历顺序等）。
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' # 设置 CUBLAS 的工作空间大小，CUBLAS 是 NVIDIA GPU 上的 BLAS 库。
def eval_epoch_train_finetune(dataset, model, device):
    all_logits = []
    all_labels = []
    all_feats = []
    all_gru_scores = []

    with torch.no_grad():
        model.eval()

        for batch_idx, batch_sample in enumerate(dataset):
            # ===== 1. 读取当前 batch =====
            input_nodes_feature = batch_sample["input_nodes_feature"]   # [B, Nmax, Dv]
            input_edges_feature = batch_sample["input_edges_feature"]   # [B, Emax, De]
            labels = batch_sample["labels"]                             # [B]
            Tmats = batch_sample["Tmats"]                               # [B, Nmax, Emax]
            adjs = batch_sample["adjs"]                                 # [B, Nmax, Nmax]
            eadjs = batch_sample["eadjs"]                               # [B, Emax, Emax]
            edge_mask = batch_sample["edge_mask"]                       # [B, Emax]
            num_nodes = batch_sample["num_nodes"]                       # [B]
            num_edges = batch_sample["num_edges"]                       # [B]
            target_edge_index = batch_sample["target_edge_index"]       # [B]
            gru_score = batch_sample["gru_score"]                       # [B]

            # ===== 2. 稀疏转 dense，并搬到 device =====
            def ensure_dense(tensor):
                if hasattr(tensor, "is_sparse") and tensor.is_sparse:
                    return tensor.to_dense()
                return tensor

            input_nodes_feature = ensure_dense(input_nodes_feature.to(device))
            input_edges_feature = ensure_dense(input_edges_feature.to(device))
            Tmats = ensure_dense(Tmats.to(device))
            adjs = ensure_dense(adjs.to(device))
            eadjs = ensure_dense(eadjs.to(device))
            edge_mask = ensure_dense(edge_mask.to(device))

            num_nodes = num_nodes.to(device)
            num_edges = num_edges.to(device)
            target_edge_index = target_edge_index.to(device)
            gru_score = gru_score.to(device)

            # ===== 3. 前向推理 =====
            logits, feat = model(
                input_nodes_feature,     # H_v
                input_edges_feature,     # H_e
                input_edges_feature,     # H_e_pad（兼容旧接口，当前不再实际使用）
                eadjs,                   # adj_e
                adjs,                    # adj_v
                Tmats,                   # T
                edge_mask,               # mask
                num_nodes=num_nodes,
                num_edges=num_edges,
                target_edge_index=target_edge_index
            )

            # ===== 4. 保存结果 =====
            y = labels.to(device).long()

            logits_cpu = logits.detach().cpu().numpy()      # [B, num_classes]
            feat_cpu = feat.detach().cpu().numpy()          # [B, feat_dim]
            y_cpu = y.detach().cpu().numpy()                # [B]
            gru_cpu = gru_score.detach().cpu().numpy()      # [B]

            all_logits.append(logits_cpu)
            all_labels.append(y_cpu)
            all_feats.append(feat_cpu)
            all_gru_scores.append(gru_cpu)

    all_logits_array = np.concatenate(all_logits, axis=0)        # [N, num_classes]
    all_labels_array = np.concatenate(all_labels, axis=0)        # [N]
    all_feats_array = np.concatenate(all_feats, axis=0)          # [N, feat_dim]
    all_gru_scores_array = np.concatenate(all_gru_scores, axis=0)  # [N]

    return all_logits_array, all_labels_array, all_feats_array, all_gru_scores_array

def eval_epoch_finetune_one(
    dataset,
    model,
    model_if_logit,
    model_if_feat,
    model_if_logit_gru,
    model_if_feat_gru,
    device
):
    all_logits = []
    all_labels = []
    all_feats = []
    all_gru_scores = []

    with torch.no_grad():
        model.eval()

        for batch_idx, batch_sample in enumerate(dataset):
            # ===== 1. 读取当前 batch =====
            input_nodes_feature = batch_sample["input_nodes_feature"]   # [B, Nmax, Dv]
            input_edges_feature = batch_sample["input_edges_feature"]   # [B, Emax, De]
            labels = batch_sample["labels"]                             # [B]
            Tmats = batch_sample["Tmats"]                               # [B, Nmax, Emax]
            adjs = batch_sample["adjs"]                                 # [B, Nmax, Nmax]
            eadjs = batch_sample["eadjs"]                               # [B, Emax, Emax]
            edge_mask = batch_sample["edge_mask"]                       # [B, Emax]
            num_nodes = batch_sample["num_nodes"]                       # [B]
            num_edges = batch_sample["num_edges"]                       # [B]
            target_edge_index = batch_sample["target_edge_index"]       # [B]
            gru_score = batch_sample["gru_score"]                       # [B]

            # ===== 2. 稀疏转 dense，并移动到 device =====
            def ensure_dense(tensor):
                if hasattr(tensor, "is_sparse") and tensor.is_sparse:
                    return tensor.to_dense()
                return tensor

            input_nodes_feature = ensure_dense(input_nodes_feature.to(device))
            input_edges_feature = ensure_dense(input_edges_feature.to(device))
            Tmats = ensure_dense(Tmats.to(device))
            adjs = ensure_dense(adjs.to(device))
            eadjs = ensure_dense(eadjs.to(device))
            edge_mask = ensure_dense(edge_mask.to(device))

            num_nodes = num_nodes.to(device)
            num_edges = num_edges.to(device)
            target_edge_index = target_edge_index.to(device)
            gru_score = gru_score.to(device)

            # ===== 3. 前向推理 =====
            logits, feat = model(
                input_nodes_feature,     # H_v
                input_edges_feature,     # H_e
                input_edges_feature,     # H_e_pad（兼容旧接口）
                eadjs,                   # adj_e
                adjs,                    # adj_v
                Tmats,                   # T
                edge_mask,               # mask
                num_nodes=num_nodes,
                num_edges=num_edges,
                target_edge_index=target_edge_index
            )

            # ===== 4. 收集结果 =====
            y = labels.to(device).long()

            all_logits.append(logits.detach().cpu().numpy())        # [B, 2]
            all_labels.append(y.detach().cpu().numpy())             # [B]
            all_feats.append(feat.detach().cpu().numpy())           # [B, feat_dim]
            all_gru_scores.append(gru_score.detach().cpu().numpy()) # [B]

    # =========================
    # 汇总
    # =========================
    all_logits = np.concatenate(all_logits, axis=0)
    all_logits = np.nan_to_num(all_logits, nan=0.0, posinf=1e6, neginf=-1e6)

    all_labels = np.concatenate(all_labels, axis=0).astype(int)

    all_feats = np.concatenate(all_feats, axis=0)
    all_feats = np.nan_to_num(all_feats, nan=0.0, posinf=1e6, neginf=-1e6)

    all_gru_scores = np.concatenate(all_gru_scores, axis=0)
    all_gru_scores = np.nan_to_num(all_gru_scores, nan=0.0, posinf=1e6, neginf=-1e6)

    y_true = all_labels.astype(int)

    # =========================
    # 一些公共辅助函数
    # =========================
    def eval_direct_logits(logits_array, y_true):
        """
        直接评估 Transformer 输出 logits。
        默认二分类：第 1 列视为 anomaly 分数。
        """
        y_pred = np.argmax(logits_array, axis=1)

        if logits_array.shape[1] > 1:
            score = logits_array[:, 1]
        else:
            score = logits_array[:, 0]

        auc = roc_auc_score(y_true, score)
        rec = average_precision_score(y_true, score)
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

        return auc, acc, prec, rec, f1

    def eval_if_model(if_model, x_input, y_true):
        """
        用 IsolationForest 评估：
        - predict: 1=normal, -1=anomaly
        - decision_function: 越大越正常，因此取负号作为 anomaly score
        """
        yhat_pred = if_model.predict(x_input)
        y_pred = np.where(yhat_pred == 1, 0, 1)   # 1->normal(0), -1->anomaly(1)

        anomaly_score = -if_model.decision_function(x_input)

        auc = roc_auc_score(y_true, anomaly_score)
        rec = average_precision_score(y_true, anomaly_score)
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

        return auc, acc, prec, rec, f1

    # =========================
    # （1）Transformer 直接输出结果
    # =========================
    direct_auc, direct_acc, direct_prec, direct_rec, direct_f1 = eval_direct_logits(all_logits, y_true)

    # =========================
    # （2）Transformer 的结果 + IF
    # =========================
    logit_if_auc, logit_if_acc, logit_if_prec, logit_if_rec, logit_if_f1 = eval_if_model(
        model_if_logit, all_logits, y_true
    )

    # =========================
    # （3）Transformer 的表征 + IF
    # =========================
    feat_if_auc, feat_if_acc, feat_if_prec, feat_if_rec, feat_if_f1 = eval_if_model(
        model_if_feat, all_feats, y_true
    )

    # =========================
    # （4）GRU 结果 + Transformer 的结果 + IF
    # =========================
    all_logits_gru = np.concatenate([all_logits, all_gru_scores.reshape(-1, 1)], axis=1)
    logit_gru_if_auc, logit_gru_if_acc, logit_gru_if_prec, logit_gru_if_rec, logit_gru_if_f1 = eval_if_model(
        model_if_logit_gru, all_logits_gru, y_true
    )

    # =========================
    # （5）GRU 结果 + Transformer 的表征 + IF
    # =========================
    all_feats_gru = np.concatenate([all_feats, all_gru_scores.reshape(-1, 1)], axis=1)
    feat_gru_if_auc, feat_gru_if_acc, feat_gru_if_prec, feat_gru_if_rec, feat_gru_if_f1 = eval_if_model(
        model_if_feat_gru, all_feats_gru, y_true
    )

    return (
        direct_auc, direct_acc, direct_prec, direct_rec, direct_f1,
        logit_if_auc, logit_if_acc, logit_if_prec, logit_if_rec, logit_if_f1,
        feat_if_auc, feat_if_acc, feat_if_prec, feat_if_rec, feat_if_f1,
        logit_gru_if_auc, logit_gru_if_acc, logit_gru_if_prec, logit_gru_if_rec, logit_gru_if_f1,
        feat_gru_if_auc, feat_gru_if_acc, feat_gru_if_prec, feat_gru_if_rec, feat_gru_if_f1
    )

import torch
import torch.nn.functional as F


def bpr_loss_binary_normal_only(
    logits,
    y=None,
    idx=None,
    intra_margin=0.05,
    pseudo_margin=0.10,
    lambda_pseudo=0.30,
    hard_ratio=0.25
):
    """
    适用于“正常类为0”的二分类 BPR 损失（增强版）。

    在只能使用当前 batch logits 的前提下，加入“batch 内伪异常代理”思想：

    1) 样本内排序：
       对正常样本约束 normal_logit > abnormal_logit

    2) batch 内伪异常排序：
       - 从当前 batch 的正常样本中，
         取 anomaly score 最高的一部分作为“伪异常代理 hard negatives”
       - 取 anomaly score 最低的一部分作为“正常锚点 easy positives”
       - 约束 hard negatives 的异常分数 > easy positives 的异常分数

    参数:
        logits: [N, 2]
        y: 可选，[N]；若提供，则只在 y==0 的样本上计算
        idx: 可选；若提供，则只在 idx 指定样本上计算
        intra_margin: 样本内排序间隔
        pseudo_margin: batch 内伪异常排序间隔
        lambda_pseudo: batch 内伪异常排序项的权重
        hard_ratio: 用于构造 hard/easy 子集的比例

    返回:
        标量 loss
    """
    device = logits.device

    if logits.ndim != 2 or logits.size(1) != 2:
        raise ValueError(
            f"[BPR-normal-only] logits 应为 [N, 2]，但收到 {tuple(logits.shape)}"
        )

    N = logits.size(0)

    # ========= 处理 idx =========
    if idx is None:
        idx = torch.arange(N, device=device, dtype=torch.long)
    elif not torch.is_tensor(idx):
        idx = torch.as_tensor(idx, dtype=torch.long, device=device)
    else:
        idx = idx.to(device=device, dtype=torch.long)

    idx = idx.view(-1)

    if idx.numel() == 0:
        return logits.sum() * 0.0

    idx_min = int(idx.min().item())
    idx_max = int(idx.max().item())
    if idx_min < 0 or idx_max >= N:
        raise ValueError(
            f"[BPR-normal-only] idx 越界: idx_min={idx_min}, idx_max={idx_max}, N={N}"
        )

    # ========= 构造有效样本 mask =========
    valid_mask = torch.zeros(N, dtype=torch.bool, device=device)
    valid_mask[idx] = True

    if y is not None:
        if not torch.is_tensor(y):
            y = torch.as_tensor(y, dtype=torch.long, device=device)
        else:
            y = y.to(device=device, dtype=torch.long)

        y = y.view(-1)
        if y.numel() != N:
            raise ValueError(
                f"[BPR-normal-only] y 长度必须等于 N，收到 len(y)={y.numel()}, N={N}"
            )

        # 只在正常样本上计算
        valid_mask = valid_mask & (y == 0)

    if valid_mask.sum().item() == 0:
        return logits.sum() * 0.0

    logits_b = logits[valid_mask]   # [B, 2]

    # anomaly_score 越大，越像异常
    normal_score = logits_b[:, 0]
    abnormal_score = logits_b[:, 1]
    anomaly_score = abnormal_score - normal_score   # [B]

    # =========================================================
    # 1) 样本内排序：normal_score > abnormal_score
    #    等价于 anomaly_score 尽量小
    # =========================================================
    intra_loss = F.softplus(anomaly_score + intra_margin).mean()

    # =========================================================
    # 2) batch 内伪异常代理排序
    #    - 选 anomaly_score 高的一部分作为 hard negatives
    #    - 选 anomaly_score 低的一部分作为 easy positives
    #    - 约束 hard > easy + pseudo_margin
    # =========================================================
    B = anomaly_score.numel()

    # batch 太小时，不做这项，避免噪声太大
    if B < 4 or lambda_pseudo <= 0.0 or hard_ratio <= 0.0:
        return intra_loss

    k = max(1, int(B * hard_ratio))
    if 2 * k > B:
        k = B // 2

    if k == 0:
        return intra_loss

    # 用 detach 的分数来做 hard/easy 划分，避免排序选择本身带来不稳定梯度
    score_detached = anomaly_score.detach()
    sorted_idx = torch.argsort(score_detached)  # 从小到大

    easy_idx = sorted_idx[:k]   # 最像正常
    hard_idx = sorted_idx[-k:]  # 最像异常

    easy_score = anomaly_score[easy_idx]  # [k]
    hard_score = anomaly_score[hard_idx]  # [k]

    # 希望 hard_score > easy_score + pseudo_margin
    pseudo_diff = hard_score - easy_score - pseudo_margin
    pseudo_loss = F.softplus(-pseudo_diff).mean()

    loss = intra_loss + lambda_pseudo * pseudo_loss
    return loss