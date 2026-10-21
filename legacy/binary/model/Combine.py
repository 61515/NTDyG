import torch
import torch.nn as nn


class CombinedModel(nn.Module):
    """
    组合模型：
    1. GNN 先对整张图做边表示学习
    2. 只提取目标事件边的两类表示：
       - gnn_target_feat: GNN 输出的目标事件边表示
       - raw_target_feat: 原始目标边特征（未经过 GNN）
    3. 将这两个向量送入 TransformerBinaryClassifier 做异常检测

    说明：
    - 为了尽量兼容旧代码，forward 里保留 H_e_pad 参数，但当前版本不再使用
    """
    def __init__(self, config, GNN, Transformer):
        super(CombinedModel, self).__init__()
        self.num_classes = getattr(config, "num_classes", 2)
        self.GNN = GNN
        self.Transformer = Transformer

    def _infer_num_edges_from_mask(self, mask):
        """
        mask: [B, Emax], 0 有效，1 padding
        """
        return (mask == 0).sum(dim=1)

    def _gather_target_edge_feature(self, H_e, target_edge_index):
        """
        从原始边特征 H_e 中提取目标事件边特征。

        输入:
            H_e: [B, Emax, D]
            target_edge_index: [B]

        输出:
            raw_target_feat: [B, D]
        """
        B = H_e.shape[0]
        batch_idx = torch.arange(B, device=H_e.device)
        return H_e[batch_idx, target_edge_index, :]

    def forward(
        self,
        H_v,
        H_e,
        H_e_pad,   # 兼容旧接口，当前不再使用
        adj_e,
        adj_v,
        T,
        mask,
        num_nodes=None,
        num_edges=None,
        target_edge_index=None
    ):
        """
        输入:
            H_v: [B, Nmax, node_dim]
            H_e: [B, Emax, edge_dim]
            H_e_pad: 保留旧接口，不使用
            adj_e: [B, Emax, Emax]
            adj_v: [B, Nmax, Nmax]
            T: [B, Nmax, Emax]
            mask: [B, Emax], 0 有效，1 padding
            num_nodes: [B], 可选
            num_edges: [B], 可选
            target_edge_index: [B]，必须提供

        输出:
            output:
                二分类 logits
            feat1:
                Transformer 融合后的最终特征
        """
        del H_e_pad  # 当前版本不再需要

        if target_edge_index is None:
            raise ValueError("CombinedModel.forward 必须提供 target_edge_index。")

        if num_edges is None:
            num_edges = self._infer_num_edges_from_mask(mask)

        # 1) 从原始边特征中提取目标事件边原始特征
        raw_target_feat = self._gather_target_edge_feature(H_e, target_edge_index)  # [B, D]

        # 2) 用 GNN 提取目标事件边的图上下文表示
        _, gnn_target_feat = self.GNN(
            H_v,
            H_e,
            adj_e,
            adj_v,
            T,
            num_nodes=num_nodes,
            num_edges=num_edges,
            edge_mask=mask,
            target_edge_index=target_edge_index,
            return_target_edge=True
        )  # [B, D]

        # 3) 将两种目标边表示送入 Transformer
        output, feat1 = self.Transformer(
            raw_input=raw_target_feat,
            GNN_output=gnn_target_feat,
            mask=None
        )

        return output, feat1