import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionFusionLayer(nn.Module):
    """
    单层交叉注意力融合层：

    - Query / Key 来自 GNN 表示
    - Value 来自原始边特征

    输入:
        gnn_state: [B, 1, D]
        raw_value: [B, 1, D]

    输出:
        fused_state: [B, 1, D]
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(CrossAttentionFusionLayer, self).__init__()

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, gnn_state, raw_value):
        """
        gnn_state: [B, 1, D]
        raw_value: [B, 1, D]

        这里严格遵循你的思路：
        - q = k = gnn_state
        - v = raw_value
        """
        q = gnn_state
        k = gnn_state
        v = raw_value

        attn_out, _ = self.cross_attn(q, k, v)   # [B, 1, D]

        # 残差 + LayerNorm
        x = self.norm1(gnn_state + self.dropout1(attn_out))

        # 前馈网络
        ff = self.linear2(self.dropout(F.relu(self.linear1(x))))
        x = self.norm2(x + self.dropout2(ff))

        return x


class TransformerTargetEdgeClassifier(nn.Module):
    """
    面向目标事件边分类/排序的 Transformer 分类器。

    输入:
        1. raw_input: 原始目标边特征 [B, D]
        2. GNN_output: GNN 提取的目标事件边表示 [B, D]

    设计思路（严格按你的要求）:
        - Query / Key 使用 GNN_output
        - Value 使用 raw_input
        - 通过多层交叉注意力融合，得到目标边的融合表示
        - 最终拼接 [fused_feat, raw_input] 作为分类特征

    输出:
        logits: [B, num_classes]
        feat:   [B, 2D]
    """
    def __init__(self, config, device, hidden_size=128):
        super(TransformerTargetEdgeClassifier, self).__init__()
        self.device = device
        self.input_size = config.input_dim
        self.num_classes = config.num_classes

        self.layers = nn.ModuleList([
            CrossAttentionFusionLayer(
                d_model=self.input_size,
                nhead=config.n_heads,
                dim_feedforward=hidden_size,
                dropout=config.drop_out
            )
            for _ in range(config.n_layer)
        ])

        # 最终特征拼接后为 2D
        self.norm = nn.LayerNorm(self.input_size * 2)
        self.dropout = nn.Dropout(config.drop_out)

        self.classifier = nn.Sequential(
            nn.Linear(self.input_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(config.drop_out),
            nn.Linear(hidden_size, self.num_classes)
        )

        self.to(device)

    def forward(self, raw_input, GNN_output, mask=None):
        """
        输入:
            raw_input:  [B, D]
            GNN_output: [B, D]
            mask: 当前版本不使用，仅保留接口兼容

        输出:
            logits: [B, num_classes]
            feat:   [B, 2D]
        """
        del mask

        raw_input = raw_input.float().to(self.device)
        GNN_output = GNN_output.float().to(self.device)

        if raw_input.dim() != 2 or GNN_output.dim() != 2:
            raise ValueError(
                f"TransformerTargetEdgeClassifier 要求输入为二维张量 [B, D]，"
                f"但收到 raw_input={raw_input.shape}, GNN_output={GNN_output.shape}"
            )

        if raw_input.shape != GNN_output.shape:
            raise ValueError(
                f"raw_input 与 GNN_output 形状不一致: "
                f"{raw_input.shape} vs {GNN_output.shape}"
            )

        # 扩展成长度为 1 的“序列”
        # gnn_state 作为 Q/K
        # raw_value 作为 V
        gnn_state = GNN_output.unsqueeze(1)   # [B, 1, D]
        raw_value = raw_input.unsqueeze(1)    # [B, 1, D]

        # 多层交叉注意力融合
        x = gnn_state
        for layer in self.layers:
            x = layer(x, raw_value)

        fused_feat = x.squeeze(1)             # [B, D]

        # 为了保留原始目标边信息，最终拼接 [融合表示, 原始特征]
        feat = torch.cat([fused_feat, raw_input], dim=-1)   # [B, 2D]
        feat = self.norm(feat)
        feat = self.dropout(feat)

        logits = self.classifier(feat)        # [B, num_classes]

        return logits, feat