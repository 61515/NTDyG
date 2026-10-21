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

def alignment_infonce(a, b, tau=0.07, mask=None, symmetric=True):
        """
        InfoNCE（对比式对齐）：用 batch 内其他样本作为负样本
        a, b: (B, D) 必须 batch 对齐（第 i 个 a 对应第 i 个 b）
        tau: 温度
        mask: 可选布尔索引 (B,) 表示保留哪些对
        symmetric: 是否做双向（a->b 和 b->a），通常取 True
        """
        if mask is not None:
            a = a[mask]
            b = b[mask]
        # 如果 batch 太小（<2），返回 0
        if a.size(0) < 2:
            return torch.tensor(0.0, device=a.device)

        # L2 归一化
        a_norm = F.normalize(a, p=2, dim=1)
        b_norm = F.normalize(b, p=2, dim=1)

        # 相似度矩阵 (B, B)
        sim = torch.matmul(a_norm, b_norm.t())  # cosine 相似度（范围 -1 到 1）
        logits = sim / tau

        labels = torch.arange(a.size(0), device=a.device)  # 第 i 行的正样本索引是 i

        loss_ab = F.cross_entropy(logits, labels)  # a->b
        if symmetric:
            loss_ba = F.cross_entropy(logits.t(), labels)  # b->a
            loss = 0.5 * (loss_ab + loss_ba)
        else:
            loss = loss_ab
        return loss


def eval_epoch_train(dataset, edgelist_dataset, model, device, phase):
        all_logits = []
        all_labels = []
        all_feats = []
        all_logits_2 = []
        with torch.no_grad(): #去除梯度
            model.eval() # 设置为 eval 模式
            for batch_idx, batch_sample in enumerate(dataset):
            # for batch_sample in dataset:
                # 1. 从 batch 中取出各种输入与标签
                batch_indices = batch_sample['Timestamp_Label']
                batch_data_map = edgelist_dataset.get_train_batch(batch_indices.numpy(), device)        

                input_nodes_feature = batch_sample['input_nodes_feature']
                input_edges_feature = batch_sample['input_edges_feature']
                input_edges_pad = batch_sample['input_edges_pad']
                labels = batch_sample['labels2']
                Tmats = batch_sample['Tmats']
                adjs = batch_sample['adjs']
                eadjs = batch_sample['eadjs']
                mask_edge = batch_sample['mask_edge']
                # 2. 将所有张量移到指定 device（GPU/CPU）
                input_nodes_feature = [tensor.to(device) for tensor in input_nodes_feature]
                input_edges_feature = [tensor.to(device) for tensor in input_edges_feature]
                Tmats = [tensor.to(device) for tensor in Tmats]
                adjs = [tensor.to(device) for tensor in adjs]
                eadjs = [tensor.to(device) for tensor in eadjs]
                # 3. 前向推理
                logits, logits_2, _, feat, user_emb, item_emb, _ = model(
                        input_nodes_feature,
                        input_edges_feature,
                        input_edges_pad.to(device),
                        eadjs,
                        adjs,
                        Tmats,
                        mask_edge.to(device),
                        batch_data_map,
                        'pretrain'
                    ) # 调用模型的 forward，传入准备好的各输入
                

                # 4. 准备标签张量（浮点型）
                y = labels.to(device)
                y = y.to(torch.float32)

                logits_cpu = logits.detach().cpu().numpy()
                logits_2_cpu = logits_2.detach().cpu().numpy()
                feat_cpu = feat.detach().cpu().numpy()

                y_cpu = y.detach().cpu().numpy()
                all_logits.append(logits_cpu)
                all_logits_2.append(logits_2_cpu)
                all_labels.append(y_cpu)
                all_feats.append(feat_cpu)
        all_logits_array = np.concatenate(all_logits, axis=0)  # 变成 [N, 8] 的数组
        all_labels_array = np.concatenate(all_labels, axis=0)  # 变成 [N, 8] 的数组
        all_feats_array = np.concatenate(all_feats, axis=0)
        all_logits_2_array = np.concatenate(all_logits_2, axis=0)
        return all_logits_array, all_logits_2_array, all_labels_array, all_feats_array, user_emb, item_emb

def eval_epoch_train_finetune(dataset, model, device):
        all_logits = []
        all_labels = []
        all_feats = []
        with torch.no_grad(): #去除梯度
            model.eval() # 设置为 eval 模式
            for batch_idx, batch_sample in enumerate(dataset):
            # for batch_sample in dataset:
                # 1. 从 batch 中取出各种输入与标签
                batch_indices = batch_sample['Timestamp_Label']
                input_nodes_feature = batch_sample['input_nodes_feature']
                input_edges_feature = batch_sample['input_edges_feature']
                input_edges_pad = batch_sample['input_edges_pad']
                labels = batch_sample['labels2']
                Tmats = batch_sample['Tmats']
                adjs = batch_sample['adjs']
                eadjs = batch_sample['eadjs']
                mask_edge = batch_sample['mask_edge']
                # 2. 将所有张量移到指定 device（GPU/CPU）
                input_nodes_feature = [tensor.to(device) for tensor in input_nodes_feature]
                input_edges_feature = [tensor.to(device) for tensor in input_edges_feature]
                Tmats = [tensor.to(device) for tensor in Tmats]
                adjs = [tensor.to(device) for tensor in adjs]
                eadjs = [tensor.to(device) for tensor in eadjs]
                # 3. 前向推理
                logits, feat = model(
                        input_nodes_feature,
                        input_edges_feature,
                        input_edges_pad.to(device),
                        eadjs,
                        adjs,
                        Tmats,
                        mask_edge.to(device)

                    ) # 调用模型的 forward，传入准备好的各输入
                

                # 4. 准备标签张量（浮点型）
                y = labels.to(device)
                y = y.to(torch.float32)

                logits_cpu = logits.detach().cpu().numpy()

                feat_cpu = feat.detach().cpu().numpy()

                y_cpu = y.detach().cpu().numpy()
                all_logits.append(logits_cpu)


                all_labels.append(y_cpu)
                all_feats.append(feat_cpu)
        all_logits_array = np.concatenate(all_logits, axis=0)  # 变成 [N, 8] 的数组
        # all_logits_all_array = np.concatenate(all_logits_all, axis=0)  # 变成 [N, 8] 的数组

        all_labels_array = np.concatenate(all_labels, axis=0)  # 变成 [N, 8] 的数组
        all_feats_array = np.concatenate(all_feats, axis=0)
        return all_logits_array, all_labels_array, all_feats_array


def eval_epoch(dataset, edgelist_dataset, user_emb, item_emb, model, modelx, modelx1, modelx2, modelx3, device, phase):
        m_loss = []
        all_logits = []
        all_labels = []
        all_yhat = []
        all_logits_2 = []
        all_yhat2 = []
        all_feats = []
        with torch.no_grad(): #去除梯度
            model.eval() # 设置为 eval 模式
            for batch_idx, batch_sample in enumerate(dataset):
            # for batch_sample in dataset:
                # 1. 从 batch 中取出各种输入与标签
                batch_indices = batch_sample['Timestamp_Label']
                batch_data_map = edgelist_dataset.edgelist_test[batch_indices.numpy()]    
                batch_data_map = batch_data_map[:,0]       
                user_batch = torch.LongTensor(np.array(batch_data_map)).to(device)
                u_g_embeddings = user_emb[user_batch]
                batch_pred = torch.matmul(u_g_embeddings, item_emb.t())
                batch_pred_cpu = batch_pred.detach().cpu().numpy()
                all_logits_2.append(batch_pred_cpu)


                input_nodes_feature = batch_sample['input_nodes_feature']
                input_edges_feature = batch_sample['input_edges_feature']
                input_edges_pad = batch_sample['input_edges_pad']
                labels = batch_sample['labels2']
                Tmats = batch_sample['Tmats']
                adjs = batch_sample['adjs']
                eadjs = batch_sample['eadjs']
                mask_edge = batch_sample['mask_edge']
                # 2. 将所有张量移到指定 device（GPU/CPU）
                input_nodes_feature = [tensor.to(device) for tensor in input_nodes_feature]
                input_edges_feature = [tensor.to(device) for tensor in input_edges_feature]
                Tmats = [tensor.to(device) for tensor in Tmats]
                adjs = [tensor.to(device) for tensor in adjs]
                eadjs = [tensor.to(device) for tensor in eadjs]
                # 3. 前向推理
                logits, _, _, feat, _, _, _ = model(
                    input_nodes_feature,
                    input_edges_feature,
                    input_edges_pad.to(device),
                    eadjs,
                    adjs,
                    Tmats,
                    mask_edge.to(device),
                    None,
                    'train_test'
                )
                
                # 4. 准备标签张量（浮点型）
                y = labels.to(device)
                y = y.to(torch.float32)
                y_cpu = y.detach().cpu().numpy()
                all_labels.append(y_cpu)

                logits_cpu = logits.detach().cpu().numpy()
                all_logits.append(logits_cpu)

                feat_cpu = feat.detach().cpu().numpy()
                all_feats.append(feat_cpu)

            all_logits = np.concatenate(all_logits, axis=0)
            all_logits = np.nan_to_num(all_logits, nan=0.0, posinf=1e6, neginf=-1e6)
            all_logits_2 = np.concatenate(all_logits_2, axis=0)
            all_logits_2 = np.nan_to_num(all_logits_2, nan=0.0, posinf=1e6, neginf=-1e6)
            all_labels = np.concatenate(all_labels, axis=0).astype(int)
            all_feats = np.concatenate(all_feats, axis=0)
            all_feats = np.nan_to_num(all_feats, nan=0.0, posinf=1e6, neginf=-1e6)

            #结果一
            y_pred = np.argmax(all_logits, axis=1)
            y_true = all_labels.astype(int)

            unique_labels = np.unique(all_labels)
            num_classes = int(unique_labels.max())
            # 检查类别数量
            if num_classes > 1:
                y_onehot = label_binarize(all_labels, classes=np.arange(num_classes+1))
                auc1 = roc_auc_score(y_onehot, all_logits, average='micro', multi_class='ovr')
                rec1 = average_precision_score(y_true, all_logits, average='macro')
            else:
                y_onehot = label_binarize(all_labels, classes=np.arange(num_classes+1))
                auc1 = roc_auc_score(y_true, all_logits[:,1])
                rec1 = average_precision_score(y_true, all_logits[:,1])
            # 3) accuracy, precision, recall, f1（宏平均）
            acc1 = accuracy_score(y_true, y_pred)
            prec1 = precision_score(y_true, y_pred, average='macro', zero_division=0)
            # rec1 = recall_score(y_true, y_pred, average='macro', zero_division=0)
            # rec1 = average_precision_score(y_true, all_logits, average='macro')
            f11 = f1_score(y_true, y_pred, average='macro', zero_division=0)

            #结果二

            yhat = modelx.predict_proba(all_feats)

            y_pred = np.argmax(yhat, axis=1)
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
            # rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
            # rec = average_precision_score(y_true, yhat, average='macro')
            f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
            # auc = roc_auc_score(y_onehot, yhat, average='micro', multi_class='ovr')
            if num_classes > 1:
                auc = roc_auc_score(y_onehot, yhat, average='micro', multi_class='ovr')
                rec = average_precision_score(y_true, yhat, average='macro')
            else:
                auc = roc_auc_score(y_true, yhat[:,1])
                rec = average_precision_score(y_true, yhat[:,1])

            #结果三
            yhat2 = modelx1.predict_proba(all_logits)

            y_pred2 = np.argmax(yhat2, axis=1)
            acc2 = accuracy_score(y_true, y_pred2)
            prec2 = precision_score(y_true, y_pred2, average='macro', zero_division=0)
            # rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
            # rec2 = average_precision_score(y_true, yhat2, average='macro')
            f12 = f1_score(y_true, y_pred2, average='macro', zero_division=0)
            # auc2 = roc_auc_score(y_onehot, yhat2, average='micro', multi_class='ovr')
            if num_classes > 1:
                auc2 = roc_auc_score(y_onehot, yhat2, average='micro', multi_class='ovr')
                rec2 = average_precision_score(y_true, yhat2, average='macro')
            else:
                auc2 = roc_auc_score(y_true, yhat2[:,1])
                rec2 = average_precision_score(y_true, yhat2[:,1])

            #结果四
            if isinstance(all_logits, np.ndarray):
                all_logits = torch.from_numpy(all_logits)
            if isinstance(all_logits_2, np.ndarray):
                all_logits_2 = torch.from_numpy(all_logits_2)
            all_logits = all_logits.to(device)
            all_logits_2 = all_logits_2.to(device)  
            if isinstance(all_feats, np.ndarray):
                all_feats = torch.from_numpy(all_feats)
            all_feats = all_feats.to(device)

            x_combined = torch.cat([all_logits, all_logits_2], dim=1)
            X_xgb = x_combined.detach().cpu().numpy()
            yhat3 = modelx2.predict_proba(X_xgb)
            y_pred3 = np.argmax(yhat3, axis=1)
            # auc3 = roc_auc_score(y_onehot, yhat3, average='micro', multi_class='ovr')
            acc3 = accuracy_score(y_true, y_pred3)
            prec3 = precision_score(y_true, y_pred3, average='macro', zero_division=0)
            # rec3 = recall_score(y_true, y_pred3, average='macro', zero_division=0)
            # rec3 = average_precision_score(y_true, yhat3, average='macro')
            f13 = f1_score(y_true, y_pred3, average='macro', zero_division=0)
            if num_classes > 1:
                auc3 = roc_auc_score(y_onehot, yhat3, average='micro', multi_class='ovr')
                rec3 = average_precision_score(y_true, yhat3, average='macro')
            else:
                auc3 = roc_auc_score(y_true, yhat3[:,1])
                rec3 = average_precision_score(y_true, yhat3[:,1])

            x_combined2 = torch.cat([all_feats, all_logits_2], dim=1)
            X_xgb2 = x_combined2.detach().cpu().numpy()
            yhat4 = modelx3.predict_proba(X_xgb2)
            y_pred4 = np.argmax(yhat4, axis=1)
            acc4 = accuracy_score(y_true, y_pred4)
            prec4 = precision_score(y_true, y_pred4, average='macro', zero_division=0)
            # rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
            # rec4 = average_precision_score(y_true, yhat4, average='macro')
            f14 = f1_score(y_true, y_pred4, average='macro', zero_division=0)
            # auc4 = roc_auc_score(y_onehot, yhat4, average='micro', multi_class='ovr')

            if num_classes > 1:
                auc4 = roc_auc_score(y_onehot, yhat4, average='micro', multi_class='ovr')
                rec4 = average_precision_score(y_true, yhat4, average='macro')
            else:
                auc4 = roc_auc_score(y_true, yhat4[:,1])
                rec4 = average_precision_score(y_true, yhat4[:,1])

        return auc, acc, prec, rec, f1, auc1, acc1, prec1, rec1, f11, auc2, acc2, prec2, rec2, f12, auc3, acc3, prec3, rec3, f13, auc4, acc4, prec4, rec4, f14

def eval_epoch_finetune(dataset, edgelist_dataset, user_emb, item_emb, model, modelx, modelx1, modelx2, modelx3, device, phase):
        m_loss = []
        all_logits = []
        all_labels = []
        all_yhat = []
        all_logits_2 = []
        all_yhat2 = []
        all_feats = []
        all_logits_all = []
        with torch.no_grad(): #去除梯度
            model.eval() # 设置为 eval 模式
            for batch_idx, batch_sample in enumerate(dataset):
            # for batch_sample in dataset:
                # 1. 从 batch 中取出各种输入与标签
                batch_indices = batch_sample['Timestamp_Label']
                batch_data_map = edgelist_dataset.edgelist_test[batch_indices.numpy()]    
                batch_data_map = batch_data_map[:,0]       
                user_batch = torch.LongTensor(np.array(batch_data_map)).to(device)
                u_g_embeddings = user_emb[user_batch]
                batch_pred = torch.matmul(u_g_embeddings, item_emb.t())
                batch_pred_cpu = batch_pred.detach().cpu().numpy()
                all_logits_2.append(batch_pred_cpu)


                input_nodes_feature = batch_sample['input_nodes_feature']
                input_edges_feature = batch_sample['input_edges_feature']
                input_edges_pad = batch_sample['input_edges_pad']
                labels = batch_sample['labels2']
                Tmats = batch_sample['Tmats']
                adjs = batch_sample['adjs']
                eadjs = batch_sample['eadjs']
                mask_edge = batch_sample['mask_edge']
                # 2. 将所有张量移到指定 device（GPU/CPU）
                input_nodes_feature = [tensor.to(device) for tensor in input_nodes_feature]
                input_edges_feature = [tensor.to(device) for tensor in input_edges_feature]
                Tmats = [tensor.to(device) for tensor in Tmats]
                adjs = [tensor.to(device) for tensor in adjs]
                eadjs = [tensor.to(device) for tensor in eadjs]
                # 3. 前向推理
                logits, _, _, feat, _, _, logits_all = model(
                    input_nodes_feature,
                    input_edges_feature,
                    input_edges_pad.to(device),
                    eadjs,
                    adjs,
                    Tmats,
                    mask_edge.to(device),
                    batch_pred,
                    'train_test'
                )
                
                # 4. 准备标签张量（浮点型）
                y = labels.to(device)
                y = y.to(torch.float32)
                y_cpu = y.detach().cpu().numpy()
                all_labels.append(y_cpu)

                logits_cpu = logits.detach().cpu().numpy()
                all_logits.append(logits_cpu)
                # logits_all_cpu = logits_all.detach().cpu().numpy()
                # all_logits_all.append(logits_all_cpu)

                feat_cpu = feat.detach().cpu().numpy()
                all_feats.append(feat_cpu)

            all_logits = np.concatenate(all_logits, axis=0)
            all_logits = np.nan_to_num(all_logits, nan=0.0, posinf=1e6, neginf=-1e6)
            # all_logits_all = np.concatenate(all_logits_all, axis=0)  # 变成 [N, 8] 的数组
            # all_logits_all = np.nan_to_num(all_logits_all, nan=0.0, posinf=1e6, neginf=-1e6)
            all_logits_2 = np.concatenate(all_logits_2, axis=0)
            all_logits_2 = np.nan_to_num(all_logits_2, nan=0.0, posinf=1e6, neginf=-1e6)
            all_labels = np.concatenate(all_labels, axis=0).astype(int)
            all_feats = np.concatenate(all_feats, axis=0)
            all_feats = np.nan_to_num(all_feats, nan=0.0, posinf=1e6, neginf=-1e6)

            #结果一
            y_pred = np.argmax(all_logits, axis=1)
            y_true = all_labels.astype(int)

            unique_labels = np.unique(all_labels)
            num_classes = int(unique_labels.max())
            # 检查类别数量
            if num_classes > 1:
                y_onehot = label_binarize(all_labels, classes=np.arange(num_classes+1))
                auc1 = roc_auc_score(y_onehot, all_logits, average='micro', multi_class='ovr')
                rec1 = average_precision_score(y_true, all_logits, average='macro')
            else:
                y_onehot = label_binarize(all_labels, classes=np.arange(num_classes+1))
                auc1 = roc_auc_score(y_true, all_logits[:,1])
                rec1 = average_precision_score(y_true, all_logits[:,1])
            # 3) accuracy, precision, recall, f1（宏平均）
            acc1 = accuracy_score(y_true, y_pred)
            prec1 = precision_score(y_true, y_pred, average='macro', zero_division=0)
            # rec1 = recall_score(y_true, y_pred, average='macro', zero_division=0)
            # rec1 = average_precision_score(y_true, all_logits, average='macro')
            f11 = f1_score(y_true, y_pred, average='macro', zero_division=0)

           #结果三

            yhat = modelx.predict_proba(all_feats)

            y_pred = np.argmax(yhat, axis=1)
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
            # rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
            # rec = average_precision_score(y_true, yhat, average='macro')
            f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
            # auc = roc_auc_score(y_onehot, yhat, average='micro', multi_class='ovr')
            if num_classes > 1:
                auc = roc_auc_score(y_onehot, yhat, average='micro', multi_class='ovr')
                rec = average_precision_score(y_true, yhat, average='macro')
            else:
                auc = roc_auc_score(y_true, yhat[:,1])
                rec = average_precision_score(y_true, yhat[:,1])
            
            #结果二
            y_pred = np.argmax(all_logits_2, axis=1)
            # auc2 = roc_auc_score(y_onehot, all_logits_2, average='micro', multi_class='ovr')
            acc2 = accuracy_score(y_true, y_pred)
            prec2 = precision_score(y_true, y_pred, average='macro', zero_division=0)
            # rec1 = recall_score(y_true, y_pred, average='macro', zero_division=0)
            # rec2 = average_precision_score(y_true, all_logits_2, average='macro')
            f12 = f1_score(y_true, y_pred, average='macro', zero_division=0)
            if num_classes > 1:
                auc2 = roc_auc_score(y_onehot, all_logits_2, average='micro', multi_class='ovr')
                rec2 = average_precision_score(y_true, all_logits_2, average='macro')
            else:
                auc2 = roc_auc_score(y_true, all_logits_2[:,1])
                rec2 = average_precision_score(y_true, all_logits_2[:,1])
            # yhat2 = modelx1.predict_proba(all_logits_all)

            # y_pred2 = np.argmax(yhat2, axis=1)
            # acc2 = accuracy_score(y_true, y_pred2)
            # prec2 = precision_score(y_true, y_pred2, average='macro', zero_division=0)
            # # rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
            # rec2 = average_precision_score(y_true, yhat2, average='macro')
            # f12 = f1_score(y_true, y_pred2, average='macro', zero_division=0)
            # auc2 = roc_auc_score(y_onehot, yhat2, average='micro', multi_class='ovr')


            #结果四
            if isinstance(all_logits, np.ndarray):
                all_logits = torch.from_numpy(all_logits)
            if isinstance(all_logits_2, np.ndarray):
                all_logits_2 = torch.from_numpy(all_logits_2)
            all_logits = all_logits.to(device)
            all_logits_2 = all_logits_2.to(device)  
            if isinstance(all_feats, np.ndarray):
                all_feats = torch.from_numpy(all_feats)
            all_feats = all_feats.to(device)

            x_combined = torch.cat([all_logits, all_logits_2], dim=1)
            X_xgb = x_combined.detach().cpu().numpy()
            yhat3 = modelx2.predict_proba(X_xgb)
            y_pred3 = np.argmax(yhat3, axis=1)
            # auc3 = roc_auc_score(y_onehot, yhat3, average='micro', multi_class='ovr')
            acc3 = accuracy_score(y_true, y_pred3)
            prec3 = precision_score(y_true, y_pred3, average='macro', zero_division=0)
            # rec3 = recall_score(y_true, y_pred3, average='macro', zero_division=0)
            # rec3 = average_precision_score(y_true, yhat3, average='macro')
            f13 = f1_score(y_true, y_pred3, average='macro', zero_division=0)
            if num_classes > 1:
                auc3 = roc_auc_score(y_onehot, yhat3, average='micro', multi_class='ovr')
                rec3 = average_precision_score(y_true, yhat3, average='macro')
            else:
                auc3 = roc_auc_score(y_true, yhat3[:,1])
                rec3 = average_precision_score(y_true, yhat3[:,1])

            x_combined2 = torch.cat([all_feats, all_logits_2], dim=1)
            X_xgb2 = x_combined2.detach().cpu().numpy()
            yhat4 = modelx3.predict_proba(X_xgb2)
            y_pred4 = np.argmax(yhat4, axis=1)
            acc4 = accuracy_score(y_true, y_pred4)
            prec4 = precision_score(y_true, y_pred4, average='macro', zero_division=0)
            # rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
            # rec4 = average_precision_score(y_true, yhat4, average='macro')
            f14 = f1_score(y_true, y_pred4, average='macro', zero_division=0)
            # auc4 = roc_auc_score(y_onehot, yhat4, average='micro', multi_class='ovr')
            if num_classes > 1:
                auc4 = roc_auc_score(y_onehot, yhat4, average='micro', multi_class='ovr')
                rec4 = average_precision_score(y_true, yhat4, average='macro')
            else:
                auc4 = roc_auc_score(y_true, yhat4[:,1])
                rec4 = average_precision_score(y_true, yhat4[:,1])

        return auc, acc, prec, rec, f1, auc1, acc1, prec1, rec1, f11, auc2, acc2, prec2, rec2, f12, auc3, acc3, prec3, rec3, f13, auc4, acc4, prec4, rec4, f14


def eval_epoch_one(dataset, edgelist_dataset, model, device, phase):
        all_logits = []
        all_labels = []

        with torch.no_grad(): #去除梯度
            model.eval() # 设置为 eval 模式
            for batch_idx, batch_sample in enumerate(dataset):
            # for batch_sample in dataset:
                # 1. 从 batch 中取出各种输入与标签

                input_nodes_feature = batch_sample['input_nodes_feature']
                input_edges_feature = batch_sample['input_edges_feature']
                input_edges_pad = batch_sample['input_edges_pad']
                labels = batch_sample['labels2']
                Tmats = batch_sample['Tmats']
                adjs = batch_sample['adjs']
                eadjs = batch_sample['eadjs']
                mask_edge = batch_sample['mask_edge']
                # 2. 将所有张量移到指定 device（GPU/CPU）
                input_nodes_feature = [tensor.to(device) for tensor in input_nodes_feature]
                input_edges_feature = [tensor.to(device) for tensor in input_edges_feature]
                Tmats = [tensor.to(device) for tensor in Tmats]
                adjs = [tensor.to(device) for tensor in adjs]
                eadjs = [tensor.to(device) for tensor in eadjs]
                # 3. 前向推理
                logits, _, _, _, _, _, _ = model(
                    input_nodes_feature,
                    input_edges_feature,
                    input_edges_pad.to(device),
                    eadjs,
                    adjs,
                    Tmats,
                    mask_edge.to(device),
                    None,
                    'train_test'
                )
                
                # 4. 准备标签张量（浮点型）
                y = labels.to(device)
                y = y.to(torch.float32)
                y_cpu = y.detach().cpu().numpy()
                all_labels.append(y_cpu)

                logits_cpu = logits.detach().cpu().numpy()
                all_logits.append(logits_cpu)


            all_logits = np.concatenate(all_logits, axis=0)
            all_logits = np.nan_to_num(all_logits, nan=0.0, posinf=1e6, neginf=-1e6)
            all_labels = np.concatenate(all_labels, axis=0).astype(int)


            #结果一
            y_pred = np.argmax(all_logits, axis=1)
            y_true = all_labels.astype(int)

            unique_labels = np.unique(all_labels)
            num_classes = int(unique_labels.max())
            # 检查类别数量
            if num_classes > 1:
                y_onehot = label_binarize(all_labels, classes=np.arange(num_classes+1))
                auc1 = roc_auc_score(y_onehot, all_logits, average='micro', multi_class='ovr')
                rec1 = average_precision_score(y_true, all_logits, average='macro')
            else:
                y_onehot = label_binarize(all_labels, classes=np.arange(num_classes+1))
                auc1 = roc_auc_score(y_true, all_logits[:,1])
                rec1 = average_precision_score(y_true, all_logits[:,1])
            # 3) accuracy, precision, recall, f1（宏平均）
            acc1 = accuracy_score(y_true, y_pred)
            prec1 = precision_score(y_true, y_pred, average='macro', zero_division=0)
            # rec1 = recall_score(y_true, y_pred, average='macro', zero_division=0)
            # rec1 = average_precision_score(y_true, all_logits, average='macro')
            f11 = f1_score(y_true, y_pred, average='macro', zero_division=0)


        return auc1, acc1, prec1, rec1, f11

def eval_epoch_finetune_one(dataset, model, modelx, modelx1, device):
        all_logits = []
        all_labels = []
        all_feats = []
        with torch.no_grad(): #去除梯度
            model.eval() # 设置为 eval 模式
            for batch_idx, batch_sample in enumerate(dataset):
            # for batch_sample in dataset:
                # 1. 从 batch 中取出各种输入与标签
                batch_indices = batch_sample['Timestamp_Label']
                input_nodes_feature = batch_sample['input_nodes_feature']
                input_edges_feature = batch_sample['input_edges_feature']
                input_edges_pad = batch_sample['input_edges_pad']
                labels = batch_sample['labels2']
                Tmats = batch_sample['Tmats']
                adjs = batch_sample['adjs']
                eadjs = batch_sample['eadjs']
                mask_edge = batch_sample['mask_edge']
                # 2. 将所有张量移到指定 device（GPU/CPU）
                input_nodes_feature = [tensor.to(device) for tensor in input_nodes_feature]
                input_edges_feature = [tensor.to(device) for tensor in input_edges_feature]
                Tmats = [tensor.to(device) for tensor in Tmats]
                adjs = [tensor.to(device) for tensor in adjs]
                eadjs = [tensor.to(device) for tensor in eadjs]
                # 3. 前向推理
                logits, feat = model(
                    input_nodes_feature,
                    input_edges_feature,
                    input_edges_pad.to(device),
                    eadjs,
                    adjs,
                    Tmats,
                    mask_edge.to(device)
                )
                
                # 4. 准备标签张量（浮点型）
                y = labels.to(device)
                y = y.to(torch.float32)
                y_cpu = y.detach().cpu().numpy()
                all_labels.append(y_cpu)

                logits_cpu = logits.detach().cpu().numpy()
                all_logits.append(logits_cpu)

                feat_cpu = feat.detach().cpu().numpy()
                all_feats.append(feat_cpu)

            all_logits = np.concatenate(all_logits, axis=0)
            all_logits = np.nan_to_num(all_logits, nan=0.0, posinf=1e6, neginf=-1e6)
            all_labels = np.concatenate(all_labels, axis=0).astype(int)
            all_feats = np.concatenate(all_feats, axis=0)
            all_feats = np.nan_to_num(all_feats, nan=0.0, posinf=1e6, neginf=-1e6)

            #结果一
            y_pred = np.argmax(all_logits, axis=1)
            y_true = all_labels.astype(int)
            num_classes = 1
            # 检查类别数量
            if num_classes > 1:
                y_onehot = label_binarize(all_labels, classes=np.arange(num_classes+1))
                auc1 = roc_auc_score(y_onehot, all_logits, average='micro', multi_class='ovr')
                rec1 = average_precision_score(y_true, all_logits, average='macro')
            else:
                y_onehot = label_binarize(all_labels, classes=np.arange(num_classes+1))
                auc1 = roc_auc_score(y_true, all_logits[:,1])
                rec1 = average_precision_score(y_true, all_logits[:,1])

            acc1 = accuracy_score(y_true, y_pred)
            prec1 = precision_score(y_true, y_pred, average='macro', zero_division=0)
            f11 = f1_score(y_true, y_pred, average='macro', zero_division=0)

            #结果二

            yhat = modelx.predict(all_feats)
            y_pred = np.where(yhat == 1, 0, 1)  # 转成 0=normal,1=anomaly
            yhat = -modelx.decision_function(all_feats)
            # y_pred = np.argmax(yhat, axis=1)
            acc2 = accuracy_score(y_true, y_pred)
            prec2 = precision_score(y_true, y_pred, average='macro', zero_division=0)
            # rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
            # rec = average_precision_score(y_true, yhat, average='macro')
            f12 = f1_score(y_true, y_pred, average='macro', zero_division=0)
            # auc = roc_auc_score(y_onehot, yhat, average='micro', multi_class='ovr')
            if num_classes > 1:
                auc2 = roc_auc_score(y_onehot, yhat, average='micro', multi_class='ovr')
                rec2 = average_precision_score(y_true, yhat, average='macro')
            else:
                auc2 = roc_auc_score(y_true, yhat)
                rec2 = average_precision_score(y_true, yhat)


            #结果三
            yhat3 = modelx1.predict(all_logits)
            y_pred3 = np.where(yhat3 == 1, 0, 1)  # 转成 0=normal,1=anomaly
            yhat3 = -modelx1.decision_function(all_logits)
            acc3 = accuracy_score(y_true, y_pred3)
            prec3 = precision_score(y_true, y_pred3, average='macro', zero_division=0)
            f13 = f1_score(y_true, y_pred3, average='macro', zero_division=0)
            if num_classes > 1:
                auc3 = roc_auc_score(y_onehot, yhat3, average='micro', multi_class='ovr')
                rec3 = average_precision_score(y_true, yhat3, average='macro')
            else:
                auc3 = roc_auc_score(y_true, yhat3)
                rec3 = average_precision_score(y_true, yhat3)

        return auc1, acc1, prec1, rec1, f11, auc2, acc2, prec2, rec2, f12, auc3, acc3, prec3, rec3, f13


def bpr_loss_multiclass(logits, y, idx, num_neg=5):
    """
    logits: (N, C)
    y: (N,) int64
    idx: 训练节点索引 (B,)
    num_neg: 每个样本采多少个负类
    """
    device = logits.device
    B = idx.numel()
    C = logits.size(1)

    pos_cls = y[idx]                           # (B,)
    pos_score = logits[idx, pos_cls]           # (B,)

    # 采样负类：简单做法是均匀采样直到 != pos
    neg_cls = torch.randint(0, C, (B, num_neg), device=device)
    # 避免采到正类
    mask = (neg_cls == pos_cls.unsqueeze(1))
    while mask.any():
        neg_cls[mask] = torch.randint(0, C, (mask.sum(),), device=device)
        mask = (neg_cls == pos_cls.unsqueeze(1))

    # 取负类分数 (B, num_neg)
    neg_score = logits[idx].gather(1, neg_cls)

    # BPR: softplus(-(pos-neg))，对多个负样本取平均
    diff = pos_score.unsqueeze(1) - neg_score
    loss = F.softplus(-diff).mean()
    return loss
def bpr_loss_multiclass_safe(logits, y, idx, num_neg=5):
    device = logits.device

    # idx -> tensor long on device
    if not torch.is_tensor(idx):
        idx = torch.as_tensor(idx, dtype=torch.long, device=device)
    else:
        idx = idx.to(device=device, dtype=torch.long)

    # y -> tensor long on device
    if not torch.is_tensor(y):
        y = torch.as_tensor(y, dtype=torch.long, device=device)
    else:
        y = y.to(device=device, dtype=torch.long)

    N = logits.size(0)
    C = logits.size(1)

    # --------- 关键检查：idx 是否越界 ----------
    if idx.numel() == 0:
        return torch.tensor(0.0, device=device)

    idx_min = int(idx.min().item())
    idx_max = int(idx.max().item())
    if idx_min < 0 or idx_max >= N:
        raise ValueError(
            f"[BPR] idx 越界: idx_min={idx_min}, idx_max={idx_max}, but N={N}. "
            f"请检查 batch_indices 的来源/是否对应 logits 的行索引。"
        )

    # 取正类
    pos_cls = y.index_select(0, idx)  # (B,)

    # --------- 关键检查：label 是否越界 ----------
    y_min = int(pos_cls.min().item())
    y_max = int(pos_cls.max().item())
    if y_min < 0 or y_max >= C:
        raise ValueError(
            f"[BPR] label 越界: y_min={y_min}, y_max={y_max}, but C={C}. "
            f"说明 y 的类别编码不在 [0, C-1] 或 logits 输出维度与类别数不一致。"
        )

    # 正类分数 (B,)
    logits_b = logits.index_select(0, idx)                 # (B, C)
    pos_score = logits_b.gather(1, pos_cls.view(-1, 1)).squeeze(1)

    # 负类采样 (B, num_neg)
    neg_cls = torch.randint(0, C, (idx.numel(), num_neg), device=device)
    mask = (neg_cls == pos_cls.unsqueeze(1))
    while mask.any():
        neg_cls[mask] = torch.randint(0, C, (mask.sum(),), device=device)
        mask = (neg_cls == pos_cls.unsqueeze(1))

    neg_score = logits_b.gather(1, neg_cls)  # (B, num_neg)

    diff = pos_score.unsqueeze(1) - neg_score
    return F.softplus(-diff).mean()