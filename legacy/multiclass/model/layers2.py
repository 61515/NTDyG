import math
import sys

import numpy as np
import torch
import time
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    def __init__(self, in_features_v, out_features_v, in_features_e, out_features_e, bias=True, node_layer=True):
        super(GraphConvolution, self).__init__()
        self.in_features_e = in_features_e
        self.out_features_e = out_features_e
        self.in_features_v = in_features_v
        self.out_features_v = out_features_v
        self.node_layer = node_layer

        if node_layer:
            # self.node_layer = True
            # 用于节点更新的权重矩阵：从 in_features_v 到 out_features_v
            # self.weight = Parameter(torch.FloatTensor(in_features_v, out_features_v))
            # # p 是一个形状 (1, in_features_e) 的可学习参数，用于将边特征 H_e 投影到标量（或多通道后取第一通道）
            # self.p = Parameter(torch.from_numpy(np.random.normal(size=(1, in_features_e))).float())
            # if bias:
            #     # 节点输出偏置
            #     self.bias = Parameter(torch.FloatTensor(out_features_v))
            # else:
            #     self.register_parameter('bias', None)
            self.weight = Parameter(torch.empty(in_features_v, out_features_v))
            self.p = Parameter(torch.empty(1, in_features_e))
            self.bias = Parameter(torch.empty(out_features_v)) if bias else None
        else:
            # # self.node_layer = False
            # # 用于边更新的权重矩阵：从 in_features_e 到 out_features_e
            # self.weight = Parameter(torch.FloatTensor(in_features_e, out_features_e))
            #  # p 是形状 (1, in_features_v)，用于将节点特征 H_v 投影到标量
            # self.p = Parameter(torch.from_numpy(np.random.normal(size=(1, in_features_v))).float())
            # # 在边更新时，H_v @ p.t() 得到每个节点的标量，用于将节点信息通过 T.t() 聚合到边。
            # if bias:
            #     self.bias = Parameter(torch.FloatTensor(out_features_e))
            # else:
            #     self.register_parameter('bias', None)
            self.weight = Parameter(torch.empty(in_features_e, out_features_e))
            self.p = Parameter(torch.empty(1, in_features_v))
            self.bias = Parameter(torch.empty(out_features_e)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
         # Xavier initialization for weight
        torch.nn.init.xavier_uniform_(self.weight)
        # small normal for projection
        torch.nn.init.normal_(self.p, mean=0.0, std=0.01)
        if self.bias is not None:
            fan_in = self.weight.size(0)
            bound = 1.0 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

        # # 权重和偏置初始化为均匀分布
        # stdv = 1. / math.sqrt(self.weight.size(1))
        # self.weight.data.uniform_(-stdv, stdv)

        # if self.bias is not None:
        #     self.bias.data.uniform_(-stdv, stdv)

    def forward(self, H_v, H_e, adj_e, adj_v, T):
        """
        H_v: 节点特征矩阵，形状 [num_nodes, in_features_v]
        H_e: 边特征矩阵，形状 [num_edges, in_features_e]
        adj_v: 节点邻接矩阵，形状 [num_nodes, num_nodes]
        adj_e: 边邻接矩阵，形状 [num_edges, num_edges]
        T: 节点-边关联矩阵，形状可能是 [num_nodes, num_edges] 或其转置，用于节点和边之间的消息传递
        """
        if self.node_layer:
            # project edge features to (0,1)
            score = (H_e @ self.p.t()).squeeze(-1)
            alpha = torch.sigmoid(- score) 
            # alpha = torch.sigmoid(H_e @ self.p.t()).squeeze(-1)  # [M]
            # aggregate to node space
            # 确保设备一致
            alpha = alpha.to(T.device)

            if T.is_sparse:
                T = T.coalesce()
    # 列缩放 sparse：T * diag(alpha)
                idx = T.indices()
                vals = T.values()
                col = idx[1]
                vals_scaled = vals * alpha[col]
                T_scaled = torch.sparse_coo_tensor(idx, vals_scaled, T.size(), device=T.device).coalesce()
                M = torch.sparse.mm(T_scaled, T.t())
            else:
    # dense 情况：直接按列缩放（广播）即可
    # T: (R, C), alpha: (C,)
                T_scaled = T * alpha.view(1, -1)
                M = T_scaled @ T.t()
            # M = torch.spmm(T, torch.diag(alpha)) @ T.t()
            # self-loop + neighbor
            I = torch.eye(M.size(0), device=M.device)
            A = I + (1 - I) * M
            # elementwise mask by adj_v
            A = A * adj_v
            # row-normalize A
            row_sum = A.sum(dim=1, keepdim=True).to_dense()   # 强制转为 dense
            D_inv   = 1.0 / (row_sum + 1e-6)
            # D_inv = 1.0 / (A.sum(dim=1, keepdim=True) + 1e-6)
            A = D_inv * A
            # update node features
            support = H_v @ self.weight
            out = A @ support
            if self.bias is not None:
                out = out + self.bias
            # out = F.dropout(out, self.dropout, training=self.training)
            return out, H_e
            #  # ——节点更新分支——
            # # 1. 计算边投影到标量后的影响： H_e @ p.t() 结果为 [num_edges, 1]
            # #    (H_e @ p.t()).t()[0] 得到长度 num_edges 的向量
            # #    构造 diag 矩阵后用 T 和 T.t() 聚合回节点空间
            # # multiplier1 = torch.spmm(T, torch.diag((H_e @ self.p.t()).t()[0])) @ T.to_dense().t()
            # multiplier1 = torch.spmm(T, torch.diag((H_e @ self.p.t()).t()[0])) @ T.t()
            # # mask1: 单位矩阵，用于后面在对角位置添加 1，确保自环或自身特征保留
            # mask1 = torch.eye(multiplier1.shape[0])
            # mask1 = mask1.to(H_e.device)
            # M1 = mask1 * torch.ones(multiplier1.shape[0]).to(H_e.device) + (1. - mask1)*multiplier1
            # # adjusted_A = torch.mul(M1, adj_v.to_dense())
            #  # adjusted_A: 将原始节点邻接 adj_v 与 M1 逐元素相乘，得到加权邻接
            # adjusted_A = torch.mul(M1, adj_v)
            # # to avoid missing feature's influence, we don't normalize the A
            #  # 2. 矩阵乘： adjusted_A @ (H_v @ weight)
            # output = torch.mm(adjusted_A, torch.mm(H_v, self.weight))
            # if self.bias is not None:
            #     ret = output + self.bias
            # # 返回更新后的节点特征 ret，以及原始边特征 H_e（边特征不变）
            # return ret, H_e

        else:
             # project node features to (0,1)
            score = (H_v @ self.p.t()).squeeze(-1)
            beta = torch.sigmoid(- score) 

            # beta = torch.sigmoid(H_v @ self.p.t()).squeeze(-1)  # [N]
            M = torch.spmm(T.t(), torch.diag(beta)) @ T
            I = torch.eye(M.size(0), device=M.device)
            Ae = I + (1 - I) * M
            Ae = Ae * (adj_e.to_dense() if adj_e.is_sparse else adj_e)
            # row-normalize
            row_sum = Ae.sum(dim=1, keepdim=True)
            if row_sum.is_sparse:
                row_sum = row_sum.to_dense()
            D_inv = 1.0 / (row_sum + 1e-6)
            # D_inv = 1.0 / (Ae.sum(dim=1, keepdim=True) + 1e-6)
            Ae = D_inv * Ae
            # sparse convert
            indices = Ae.nonzero().t()
            values = Ae[indices[0], indices[1]]
            Ae_sparse = torch.sparse_coo_tensor(indices, values, Ae.size(), device=Ae.device)
            # update edge features
            support = H_e @ self.weight
            out = torch.sparse.mm(Ae_sparse, support)
            if self.bias is not None:
                out = out + self.bias
            # out = F.dropout(out, self.dropout, training=self.training)
            return H_v, out
            # # multiplier2 = torch.spmm(T.t(), torch.diag((H_v @ self.p.t()).t()[0])) @ T.to_dense()
            # # 在边更新时，H_v @ p.t() 得到每个节点的标量，用于将节点信息通过 T.t() 聚合到边。
            # # 1. 类似上面逻辑，用节点特征投影来影响边： H_v @ p.t() → diag，聚合到边空间
            # multiplier2 = torch.spmm(T.t(), torch.diag((H_v @ self.p.t()).t()[0])) @ T
            # mask2 = torch.eye(multiplier2.shape[0])
            # mask2 = mask2.to(H_e.device)
            # # adjusted_A: 将原始边邻接 adj_e 与 M3 相乘，得到加权邻接
            # M3 = mask2 * torch.ones(multiplier2.shape[0]).to(H_e.device) + (1. - mask2)*multiplier2
            
            # # --- 这里假设 adj_e 是稀疏 COO 格式 ---
            # # 如果 adj_e 不是稀疏，需要先转换： adj_e = adj_e.to_sparse_coo()
            # if not adj_e.is_sparse:
            #     adj_e = adj_e.to_sparse_coo()
            
            #  # 1. 对稀疏邻接先 coalesce，获得 indices 和 values
            # adj_e = adj_e.coalesce()  # 确保已合并重复索引 :contentReference[oaicite:1]{index=1}
            # indices = adj_e.indices()  # [2, nnz]
            # values = adj_e.values()    # [nnz]
            # # 2. 在非零条目上做加权：new_values = values * M3[row, col]
            # row_idx = indices[0]
            # col_idx = indices[1]
            # # 从 M3 中取对应位置的权重
            # weight_on_edges = M3[row_idx, col_idx]  # shape [nnz]
            # weighted_values = values * weight_on_edges
            # # 3. 对加权后的稀疏值做列归一化：对每一列 j，values[i] /= max_{i in col j}(weighted_values)
            # num_cols = adj_e.size(1)
            # # 计算每列最大值
            # max_per_col = torch.zeros(num_cols, device=weighted_values.device)
            # # scatter_reduce 需要 PyTorch 1.12+，reduce='amax'
            # max_per_col.scatter_reduce_(0, col_idx, weighted_values, reduce='amax')
            # # 避免除以 0
            # max_per_col = torch.where(max_per_col == 0, torch.ones_like(max_per_col), max_per_col)
            # # 归一化
            # norm_values = weighted_values / max_per_col[col_idx]

            # # 4. 重建归一化后的稀疏 COO 张量
            # normalized_adjusted_A = torch.sparse_coo_tensor(
            #     indices, norm_values, size=adj_e.shape,
            #     device=weighted_values.device
            # ).coalesce()

            # # 5. 用稀疏-稠密乘法计算输出：normalized_adjusted_A @ (H_e @ weight)
            # # 注意：PyTorch 2.x 可使用 torch.sparse.mm 或 torch.sparse.matmul
            # # 假设你希望输出 dense，下面用 torch.sparse.mm
            # support = torch.mm(H_e, self.weight)  # [num_edges, out_features]
            # output = torch.sparse.mm(normalized_adjusted_A, support)  # [num_edges, out_features]

            # if self.bias is not None:
            #     ret = output + self.bias
            # else:
            #     ret = output

            # # # adjusted_A = torch.mul(M3, adj_e.to_dense())
            # # adjusted_A = torch.mul(M3, adj_e)
            
            
            # # # 归一化：每列除以该列最大值，防止数值过大
            # # # 目标是按列做 max 归一化：adjusted_A[i,j] /= max_j
            # # coalesced = adjusted_A.coalesce()
            # # indices = coalesced.indices()   # shape [2, nnz]
            # # values = coalesced.values()     # shape [nnz]
            # # row_idx = indices[0]
            # # col_idx = indices[1]
            # # num_cols = adjusted_A.size(1)
            # # # 在 GPU 上用 scatter_reduce 计算每列最大值
            # # max_per_col = torch.zeros(num_cols, device=values.device)
            # # # PyTorch 1.12+ 支持 scatter_reduce；reduce='amax' 表示取最大
            # # max_per_col.scatter_reduce_(0, col_idx, values, reduce='amax')
            # # # 避免除以 0（若某列全为 0，可设 max=1）
            # # max_per_col = torch.where(max_per_col == 0, torch.ones_like(max_per_col), max_per_col)
            # # # 归一化 values
            # # norm_values = values / max_per_col[col_idx]
            # # # 重建稀疏张量
            # # normalized_adjusted_A = torch.sparse_coo_tensor(indices, norm_values, adjusted_A.shape, device=values.device)
            # # # normalized_adjusted_A = adjusted_A / adjusted_A.max(0, keepdim=True)[0]
            # # # 矩阵乘： normalized_adjusted_A @ (H_e @ weight)
            # # output = torch.mm(normalized_adjusted_A, torch.mm(H_e, self.weight))
            # # if self.bias is not None:
            # #     ret = output + self.bias
            # # # 返回原始节点特征 H_v（不变）以及更新后的边特征 ret
            # return H_v, ret

    # def __repr__(self):
    #     return self.__class__.__name__ + ' (' \
    #            + str(self.in_features) + ' -> ' \
    #            + str(self.out_features) + ')'
    def __repr__(self):
        t = 'NodeLayer' if self.node_layer else 'EdgeLayer'
        return f"GraphConvolution({t}: v {self.in_features_v}->{self.out_features_v}, e {self.in_features_e}->{self.out_features_e})"

