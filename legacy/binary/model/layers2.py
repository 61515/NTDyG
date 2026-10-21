import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    联合节点-边更新的图卷积层。

    参数顺序:
        in_features_v, out_features_v, in_features_e, out_features_e

    说明:
        - node_layer=True: 主要更新节点表示，边表示保持原样返回
        - node_layer=False: 主要更新边表示，节点表示保持原样返回
    """
    def __init__(self, in_features_v, out_features_v, in_features_e, out_features_e,
                 bias=True, node_layer=True):
        super(GraphConvolution, self).__init__()

        self.in_features_e = in_features_e
        self.out_features_e = out_features_e
        self.in_features_v = in_features_v
        self.out_features_v = out_features_v
        self.node_layer = node_layer

        if node_layer:
            # 节点更新层
            self.weight = Parameter(torch.empty(in_features_v, out_features_v))
            self.p = Parameter(torch.empty(1, in_features_e))
            self.bias = Parameter(torch.empty(out_features_v)) if bias else None
        else:
            # 边更新层
            self.weight = Parameter(torch.empty(in_features_e, out_features_e))
            self.p = Parameter(torch.empty(1, in_features_v))
            self.bias = Parameter(torch.empty(out_features_e)) if bias else None

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.normal_(self.p, mean=0.0, std=0.01)

        if self.bias is not None:
            fan_in = self.weight.size(0)
            bound = 1.0 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def _to_dense_float(self, x, device):
        """
        将输入统一转为当前 device 上的 dense float tensor。
        """
        if isinstance(x, torch.Tensor):
            if x.is_sparse:
                x = x.to_dense()
            return x.float().to(device)
        raise TypeError(f"不支持的输入类型: {type(x)}")

    def forward(self, H_v, H_e, adj_e, adj_v, T):
        """
        输入:
            H_v: [num_nodes, in_features_v]
            H_e: [num_edges, in_features_e]
            adj_e: [num_edges, num_edges]
            adj_v: [num_nodes, num_nodes]
            T: [num_nodes, num_edges]

        输出:
            node_layer=True  -> (H_v_new, H_e)
            node_layer=False -> (H_v, H_e_new)
        """
        device = H_v.device

        H_v = self._to_dense_float(H_v, device)
        H_e = self._to_dense_float(H_e, device)
        adj_e = self._to_dense_float(adj_e, device)
        adj_v = self._to_dense_float(adj_v, device)
        T = self._to_dense_float(T, device)

        if self.node_layer:
            # =========================
            # 节点更新
            # =========================
            # 用边特征生成边权重 alpha
            score = (H_e @ self.p.t()).squeeze(-1)       # [E]
            alpha = torch.sigmoid(-score)                # [E]

            # 按列缩放 T（列对应边）
            T_scaled = T * alpha.view(1, -1)             # [N, E]

            # 节点-节点消息矩阵
            M = T_scaled @ T.t()                         # [N, N]

            I = torch.eye(M.size(0), device=device, dtype=M.dtype)
            A = I + (1.0 - I) * M

            # 用原始节点邻接约束
            A = A * adj_v

            # 行归一化
            row_sum = A.sum(dim=1, keepdim=True)
            A = A / (row_sum + 1e-6)

            support = H_v @ self.weight                  # [N, out_features_v]
            out = A @ support                            # [N, out_features_v]

            if self.bias is not None:
                out = out + self.bias

            return out, H_e

        else:
            # =========================
            # 边更新
            # =========================
            # 用节点特征生成节点权重 beta
            score = (H_v @ self.p.t()).squeeze(-1)       # [N]
            beta = torch.sigmoid(-score)                 # [N]

            # 按行缩放 T（行对应节点）
            T_scaled = T * beta.view(-1, 1)              # [N, E]

            # 边-边消息矩阵
            M = T.t() @ T_scaled                         # [E, E]

            I = torch.eye(M.size(0), device=device, dtype=M.dtype)
            Ae = I + (1.0 - I) * M

            # 用原始边邻接约束
            Ae = Ae * adj_e

            # 行归一化
            row_sum = Ae.sum(dim=1, keepdim=True)
            Ae = Ae / (row_sum + 1e-6)

            support = H_e @ self.weight                  # [E, out_features_e]
            out = Ae @ support                           # [E, out_features_e]

            if self.bias is not None:
                out = out + self.bias

            return H_v, out

    def __repr__(self):
        t = 'NodeLayer' if self.node_layer else 'EdgeLayer'
        return (
            f"GraphConvolution({t}: "
            f"v {self.in_features_v}->{self.out_features_v}, "
            f"e {self.in_features_e}->{self.out_features_e})"
        )