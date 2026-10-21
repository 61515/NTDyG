import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers2 import GraphConvolution


class CensNet(nn.Module):
    """
    结合四层 GraphConvolution 的图神经网络。

    调用保持不变：
        GNN = CensNet(16, args.input_dim, args.drop_out)

    其中：
        vfeat = 节点特征维度（例如 16）
        nfeat = 边特征维度（例如 args.input_dim = 52）

    当前结构：
        第1层：边引导节点更新
        第2层：节点引导边更新
        第3层：边引导节点更新
        第4层：节点引导边更新
    """
    def __init__(self, vfeat, nfeat, dropout):
        super(CensNet, self).__init__()

        # 第一轮交互
        self.gc1 = GraphConvolution(
            in_features_v=vfeat,
            out_features_v=vfeat,
            in_features_e=nfeat,
            out_features_e=nfeat,
            node_layer=True
        )
        self.gc2 = GraphConvolution(
            in_features_v=vfeat,
            out_features_v=vfeat,
            in_features_e=nfeat,
            out_features_e=nfeat,
            node_layer=False
        )

        # # 第二轮交互（新增）
        # self.gc3 = GraphConvolution(
        #     in_features_v=vfeat,
        #     out_features_v=vfeat,
        #     in_features_e=nfeat,
        #     out_features_e=nfeat,
        #     node_layer=True
        # )
        # self.gc4 = GraphConvolution(
        #     in_features_v=vfeat,
        #     out_features_v=vfeat,
        #     in_features_e=nfeat,
        #     out_features_e=nfeat,
        #     node_layer=False
        # )

        self.dropout = dropout
        self.node_dim = vfeat
        self.edge_dim = nfeat

    def _forward_single_graph(self, X, Z, adj_e, adj_v, T):
        """
        单个图前向传播。

        输入:
            X: [N, node_dim]
            Z: [E, edge_dim]
            adj_e: [E, E]
            adj_v: [N, N]
            T: [N, E]

        输出:
            X: [N, node_dim]
            Z: [E, edge_dim]
        """
        X = X.float()
        Z = Z.float()
        adj_e = adj_e.float()
        adj_v = adj_v.float()
        T = T.float()

        # ===== 第一轮：边 -> 节点 =====
        X, Z = self.gc1(X, Z, adj_e, adj_v, T)
        X = F.relu(X)
        Z = F.relu(Z)
        X = F.dropout(X, self.dropout, training=self.training)

        # ===== 第一轮：节点 -> 边 =====
        X, Z = self.gc2(X, Z, adj_e, adj_v, T)
        X = F.relu(X)
        Z = F.relu(Z)
        Z = F.dropout(Z, self.dropout, training=self.training)

        # # ===== 第二轮：边 -> 节点（新增） =====
        # X, Z = self.gc3(X, Z, adj_e, adj_v, T)
        # X = F.relu(X)
        # Z = F.relu(Z)
        # X = F.dropout(X, self.dropout, training=self.training)

        # # ===== 第二轮：节点 -> 边（新增） =====
        # X, Z = self.gc4(X, Z, adj_e, adj_v, T)
        # X = F.relu(X)
        # Z = F.relu(Z)
        # Z = F.dropout(Z, self.dropout, training=self.training)

        return X, Z

    def _infer_num_from_mask(self, mask_tensor):
        """
        根据 mask 推断真实长度。
        约定:
            0 = 有效
            1 = padding
        """
        return (mask_tensor == 0).sum(dim=1)

    def forward(
        self,
        X_list,
        Z_list,
        adj_e_list,
        adj_v_list,
        T_list,
        num_nodes=None,
        num_edges=None,
        node_mask=None,
        edge_mask=None,
        target_edge_index=None,
        return_target_edge=False
    ):
        """
        支持两种输入形式：

        模式 A：旧版 list 输入
            X_list: list of [N_i, node_dim]
            Z_list: list of [E_i, edge_dim]
            adj_e_list: list of [E_i, E_i]
            adj_v_list: list of [N_i, N_i]
            T_list: list of [N_i, E_i]

        模式 B：新版 padded batch 输入
            X_list: [B, Nmax, node_dim]
            Z_list: [B, Emax, edge_dim]
            adj_e_list: [B, Emax, Emax]
            adj_v_list: [B, Nmax, Nmax]
            T_list: [B, Nmax, Emax]

        可选输入:
            num_nodes: [B]
            num_edges: [B]
            node_mask: [B, Nmax]
            edge_mask: [B, Emax]
            target_edge_index: [B] 或 list

        返回:
            默认:
                outputs -> list，每个元素是该图所有边的表示 [E_i, edge_dim]

            若 return_target_edge=True:
                outputs, target_edge_embeddings
                其中 target_edge_embeddings 形状为 [B, edge_dim]
        """
        outputs = []
        target_outputs = []

        # ===================================
        # 模式 A：旧版 list 输入
        # ===================================
        if isinstance(X_list, (list, tuple)):
            for i, (X, Z, adj_e, adj_v, T) in enumerate(zip(X_list, Z_list, adj_e_list, adj_v_list, T_list)):
                _, Z_out = self._forward_single_graph(X, Z, adj_e, adj_v, T)
                outputs.append(Z_out)

                if return_target_edge:
                    if target_edge_index is None:
                        raise ValueError("return_target_edge=True 时必须提供 target_edge_index。")

                    tei = int(target_edge_index[i])
                    if tei < 0 or tei >= Z_out.shape[0]:
                        raise ValueError(
                            f"第 {i} 个样本的 target_edge_index 越界: {tei}, 边数={Z_out.shape[0]}"
                        )
                    target_outputs.append(Z_out[tei])

            if return_target_edge:
                target_outputs = torch.stack(target_outputs, dim=0)
                return outputs, target_outputs

            return outputs

        # ===================================
        # 模式 B：新版 batch tensor 输入
        # ===================================
        B = X_list.shape[0]

        if num_nodes is None:
            if node_mask is None:
                raise ValueError("batch 输入模式下必须提供 num_nodes 或 node_mask。")
            num_nodes = self._infer_num_from_mask(node_mask)

        if num_edges is None:
            if edge_mask is None:
                raise ValueError("batch 输入模式下必须提供 num_edges 或 edge_mask。")
            num_edges = self._infer_num_from_mask(edge_mask)

        for i in range(B):
            n = int(num_nodes[i].item())
            e = int(num_edges[i].item())

            # 裁掉 padding，仅保留真实图
            X = X_list[i, :n, :]           # [n, node_dim]
            Z = Z_list[i, :e, :]           # [e, edge_dim]
            adj_e = adj_e_list[i, :e, :e]  # [e, e]
            adj_v = adj_v_list[i, :n, :n]  # [n, n]
            T = T_list[i, :n, :e]          # [n, e]

            _, Z_out = self._forward_single_graph(X, Z, adj_e, adj_v, T)
            outputs.append(Z_out)

            if return_target_edge:
                if target_edge_index is None:
                    raise ValueError("return_target_edge=True 时必须提供 target_edge_index。")

                tei = int(target_edge_index[i].item())
                if tei < 0 or tei >= Z_out.shape[0]:
                    raise ValueError(
                        f"batch 中第 {i} 个样本的 target_edge_index 越界: {tei}, 边数={Z_out.shape[0]}"
                    )
                target_outputs.append(Z_out[tei])

        if return_target_edge:
            target_outputs = torch.stack(target_outputs, dim=0)
            return outputs, target_outputs

        return outputs