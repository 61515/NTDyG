import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomTransformerEncoder(nn.Module):
    # 构造多层 Transformer 编码器，每层使用同样的 encoder_layer 模块（即参数共享）。
    def __init__(self, encoder_layer, num_layers):
        # encoder_layer：已初始化好的 CustomTransformerEncoderLayer 实例。
        # num_layers：层数；在 ModuleList 中重复同一实例，表示多次相同权重的前向堆叠。
        super(CustomTransformerEncoder, self).__init__()
        # 用同一个 encoder_layer 实例重复 num_layers 次
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, src, raw_src, src_mask=None, src_key_padding_mask=None):
        # src：查询（Query/Key）输入，形状 (seq_len, batch, dim).
        # raw_src：供 Value 使用的输入，形状同 (seq_len, batch, dim)，通常是原始特征。
        # src_mask：可选的 attention mask，用于屏蔽序列内部某些位置（如未来位置或自定义遮挡）。
        # src_key_padding_mask：布尔矩阵 (batch, seq_len)，True 表示该位置是 padding，需要在注意力中忽略。
        output = src
        # 依次把输出传给每一层，raw_src（例如原始输入）也一并传入
        # 将 src 依次送入每层，且每层内部都用相同的参数。
        # 每层计算 layer(output, raw_src, ...)，即 Q/K 来自前一层输出，V 固定使用 raw_src，实现“查询从逐层更新”“值始终参照原始输入”的机制。
        # 最后一层的输出 (seq_len, batch, dim)。
        for layer in self.layers:
            output = layer(output, raw_src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        return output

class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(CustomTransformerEncoderLayer, self).__init__()
        # mapped_dim = 512
        # self.pre_proj = nn.Linear(d_model, mapped_dim)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model) # norm1、norm2：两层 LayerNorm，分别放在自注意力子层和前馈子层之后，用于稳定训练。
        self.norm2 = nn.LayerNorm(d_model) # Layer Normalization（层归一化）
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, raw_src, src_mask=None, src_key_padding_mask=None):
        # Q, K from src, V from raw_src
        # # Q,K 来自 src（可能是上一层输出），V 来自 raw_src（原始输入）
        # Q、K 均取自 src，V 取自 raw_src。这样每层查询上下文来源自上一层输出，但获取的信息值始终从原始输入序列抽取。
        q = k = src # 使用当前层的表示同时作为 Query 和 Key；
        v = raw_src
        #  # 多头注意力: 输出形状 (seq_len, batch, d_model)
        # src2 = self.self_attn(q, k, v, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src2 = self.self_attn(q, k, v, attn_mask=src_mask, key_padding_mask=None)[0]
        #  # 残差 + dropout + LayerNorm
        src = src + self.dropout1(src2) # 用 src + dropout1(attn_output) 实现残差连接，再做 norm1
        src = self.norm1(src)
        #   # 前馈网络：Linear->ReLU->Dropout->Linear
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class TransformerBinaryClassifier(nn.Module):
    def __init__(self, config, device, hidden_size=128):
        super(TransformerBinaryClassifier, self).__init__()
        self.device = device
        self.input_size = config.input_dim
        # 1. 定义共享参数的自定义 Transformer 编码层
        self.encoder_layer = CustomTransformerEncoderLayer(d_model=self.input_size, nhead=config.n_heads,
                                                           dropout=config.drop_out, dim_feedforward=hidden_size)
        # 2. 将上面的层重复 num_layers 次（权重共享或可深拷贝）
        self.transformer_encoder = CustomTransformerEncoder(self.encoder_layer, num_layers=config.n_layer)
        # 3. BatchNorm + Dropout + 最后的线性分类器
        self.bn = nn.BatchNorm1d(self.input_size)
        self.dropout = nn.Dropout(config.drop_out)
        self.classifier = nn.Linear(self.input_size, 1)
        self.sigmoid = nn.Sigmoid()
        # 将模型整体搬到指定 device
        self.to(device)
        self.config = config

    def forward(self, raw_input, GNN_output, mask):
        # raw_input: 原始特征序列，用作注意力的 Value。
        # GNN_output: 来自图神经网络的特征序列，用作 Query/Key。
        # mask: 布尔矩阵，True 表示该位置是 padding，应在注意力和池化时忽略。
        
        # 1. 维度变换：Transformer 要求 (seq_len, batch, dim)
        GNN_output = GNN_output.transpose(0, 1)
        raw_input = raw_input.transpose(0, 1)
        GNN_output = GNN_output.float()
        raw_input = raw_input.float()
        mask = mask.bool()
        # 2. 编码器前向：Q/K 来自 GNN_output，V 来自 raw_input
        transformer_output = self.transformer_encoder(GNN_output, raw_input, src_key_padding_mask=mask)
        # 回到 (batch, seq_len, dim)
        transformer_output = transformer_output.transpose(0, 1)
        # 3. 对 padding 位置置零，防止它们对后续做平均产生影响
        transformer_output[mask] = 0
        # 4. 按样本提取非 padding 部分
        filtered_output = [out[~m] for out, m in zip(transformer_output, mask)]
        # 5. 对每个样本的有效 token 做平均池化
        averaged_tensors = [tensor.mean(dim=0) for tensor in filtered_output]
        # 合并成 (batch_size, dim)
        mean_output = torch.stack(averaged_tensors)

        mean_output = self.bn(mean_output)

        logits = self.classifier(mean_output) # (batch_size, 1)
        logits = self.sigmoid(logits).squeeze() # (batch_size)

        return logits, mean_output

class TransformerMultiClassClassifier(nn.Module):
    def __init__(self, config, device, hidden_size=128):
        super(TransformerMultiClassClassifier, self).__init__()
        self.device = device
        self.input_size = config.input_dim
        self.num_classes = config.num_classes  # 多分类类别数


        # 重用之前的自定义 Transformer 编码层 & Encoder
        self.encoder_layer = CustomTransformerEncoderLayer(
            d_model=self.input_size,
            nhead=config.n_heads,
            dropout=config.drop_out,
            dim_feedforward=hidden_size
        )
        self.transformer_encoder = CustomTransformerEncoder(
            self.encoder_layer,
            num_layers=config.n_layer
        )

        # 池化后再归一化、Dropout
        self.bn = nn.LayerNorm(self.input_size)
        self.dropout = nn.Dropout(config.drop_out)
        # 多分类线性层，输出 num_classes 维 logits
        self.classifier = nn.Linear(self.input_size, self.num_classes)

        # 损失函数：交叉熵损失（内部含 Softmax）
        self.criterion = nn.CrossEntropyLoss()

        self.to(device)


    def forward(self, raw_input, GNN_output, mask):
        x = GNN_output.transpose(0, 1).float().to(self.device)
        raw = raw_input.transpose(0, 1).float().to(self.device)
        mask = mask.bool().to(self.device)
        out = self.transformer_encoder(
            x, raw,
            src_key_padding_mask=mask
        )  # (seq_len, batch, dim)
        out = out.transpose(0, 1)  # (batch, seq_len, dim)

        # --- 3. 屏蔽 padding，再做平均池化
        out = out.masked_fill(mask.unsqueeze(-1), 0.0)
        lengths = (~mask).sum(dim=1, keepdim=True)  # (batch,1)
        pooled = out.sum(dim=1) / lengths  # (batch, dim)

        # --- 4. LayerNorm、Dropout、线性分类
        feat = self.bn(pooled)
        feat = self.dropout(feat)
        logits = self.classifier(feat)  # (batch, num_classes)


        return logits, feat
    
class TransformerFeatureExtractor(nn.Module):
    def __init__(self, config, device, hidden_size=128):
        super().__init__()
        self.device = device
        self.input_size = config.input_dim
        self.encoder_layer = CustomTransformerEncoderLayer(
            d_model=self.input_size,
            nhead=config.n_heads,
            dropout=config.drop_out,
            dim_feedforward=hidden_size
        )
        self.transformer_encoder = CustomTransformerEncoder(
            self.encoder_layer,
            num_layers=config.n_layer
        )
        self.bn = nn.BatchNorm1d(self.input_size)
        self.to(device)
    def forward(self, raw_input, GNN_output, mask):
        # raw_input, GNN_output: (batch, seq_len, dim)
        # mask: (batch, seq_len) 布尔型
        x = GNN_output.transpose(0,1).float().to(self.device)
        raw = raw_input.transpose(0,1).float().to(self.device)
        mask = mask.bool().to(self.device)
        out = self.transformer_encoder(x, raw, src_key_padding_mask=mask)
        out = out.transpose(0,1)  # (batch, seq_len, dim)
        out[mask] = 0  # 填充位置归零
        # 对每个样本的有效部分求均值
        pooled = torch.stack([t[~m].mean(dim=0) for t,m in zip(out, mask)], dim=0)
        pooled = self.bn(pooled)  # 归一化
        return pooled
    

class TransformerClassifier(nn.Module):
    def __init__(self, config, device, hidden_size=128):
        super().__init__()
        self.device = device
        self.input_size = config.input_dim
        self.num_classes = config.num_classes  # 包括“未知”在内的类别总数
        self.encoder_layer = CustomTransformerEncoderLayer(
            d_model=self.input_size,
            nhead=config.n_heads,
            dropout=config.drop_out,
            dim_feedforward=hidden_size
        )
        self.transformer_encoder = CustomTransformerEncoder(self.encoder_layer, num_layers=config.n_layer)
        self.bn = nn.LayerNorm(self.input_size)
        self.dropout = nn.Dropout(config.drop_out)
        self.classifier = nn.Linear(self.input_size, self.num_classes)
        self.to(device)
    def forward(self, raw_input, GNN_output, mask):
        """
        raw_input: Tensor (batch, seq_len, input_dim)
        GNN_output: Tensor (batch, seq_len, input_dim)
        mask: BoolTensor (batch, seq_len) 填充掩码
        """
        x = GNN_output.transpose(0,1).to(self.device)  # 转为 (seq_len, batch, dim)
        raw = raw_input.transpose(0,1).to(self.device)
        mask = mask.bool().to(self.device)
        out = self.transformer_encoder(x, raw, src_key_padding_mask=mask)  # (seq_len, batch, dim)
        # 将 Transformer 输出转置回 PyTorch 默认的 batch-first 格式 (batch, seq_len, dim)。
        # 对于 padding 位置，将其所有通道值置为 0，防止后续统计时影响结果。
        out = out.transpose(0,1)  # (batch, seq_len, dim)
        out = out.masked_fill(mask.unsqueeze(-1), 0.0)
        # 利用掩码计算每个样本中非 padding 的 token 数量。
        lengths = (~mask).sum(dim=1, keepdim=True)
        # 对有效位置的 token 表示求和，并除以有效 token 数得到平均特征（池化）。
        pooled = out.sum(dim=1) / lengths  # 平均池化
        feat = self.bn(pooled)
        feat = self.dropout(feat)
        logits = self.classifier(feat)  # (batch, num_classes)
        return logits, feat
    
class TransformerBinaryClassifier_no(nn.Module):
    def __init__(self, config, device, hidden_size=128):
        super(TransformerBinaryClassifier_no, self).__init__()
        self.device = device
        self.input_size = config.input_dim
        # # 1. 定义共享参数的自定义 Transformer 编码层
        # self.encoder_layer = CustomTransformerEncoderLayer(d_model=self.input_size, nhead=config.n_heads,
        #                                                    dropout=config.drop_out, dim_feedforward=hidden_size)
        # # 2. 将上面的层重复 num_layers 次（权重共享或可深拷贝）
        # self.transformer_encoder = CustomTransformerEncoder(self.encoder_layer, num_layers=config.n_layer)
        # 3. BatchNorm + Dropout + 最后的线性分类器
        self.bn = nn.BatchNorm1d(self.input_size)
        self.dropout = nn.Dropout(config.drop_out)
        self.classifier = nn.Linear(self.input_size, 1)
        self.sigmoid = nn.Sigmoid()
        # 将模型整体搬到指定 device
        self.to(device)
        self.config = config

    def forward(self, raw_input, GNN_output, mask):
                # raw_input: 原始特征序列（未使用）
        # GNN_output: 来自图神经网络的特征序列，可能形状为 [seq_len, batch_size, input_dim]
        # mask: 布尔矩阵，True 表示该位置是 padding，形状 [batch_size, seq_len]

        # 如果 GNN_output 是 (seq_len, batch_size, dim)，转换到 (batch_size, seq_len, dim)
        if GNN_output.dim() == 3 and GNN_output.shape[0] == mask.shape[1]:
            GNN_output = GNN_output.transpose(0, 1)
        # 确保为 float 和 bool
        GNN_output = GNN_output.float()
        mask = mask.bool()

        # 1. 将 padding 位置置零，防止其对平均池化产生影响
        # mask 为 True 表示 padding，unsqueeze(-1) 变为 [batch_size, seq_len, 1]
        GNN_output = GNN_output.masked_fill(mask.unsqueeze(-1), 0.0)

        # 2. 计算每个样本的有效 token 数量
        valid_counts = (~mask).sum(dim=1, keepdim=True).clamp(min=1).float()  # [batch_size, 1]

        # 3. 对非 padding 部分做平均池化，得到 [batch_size, input_dim]
        sum_features = GNN_output.sum(dim=1)  # [batch_size, input_dim]
        mean_output = sum_features / valid_counts  # [batch_size, input_dim]

        # 4. BatchNorm 和 Dropout
        mean_output = self.bn(mean_output)
        mean_output = self.dropout(mean_output)

        # 5. 线性分类和 Sigmoid
        logits = self.classifier(mean_output)  # [batch_size, 1]
        logits = self.sigmoid(logits).squeeze(-1)  # [batch_size]

        return logits, mean_output
    
class LSTMClassifier(nn.Module):
    def __init__(self, config, device, hidden_size=128):
        super(LSTMClassifier, self).__init__()
        self.device = device
        self.input_size = config.input_dim
        self.hidden_size = hidden_size
        self.num_directions = 1  # 这里假设单向
        self.lstm = nn.LSTM(self.input_size, hidden_size, config.n_layer,
                            batch_first=True, 
                            bidirectional=False)
        bidirectional=False
        # direction = 2 if bidirectional else 1
        # self.fc   = nn.Linear(hidden_size * direction, num_classes)

        self.bn = nn.BatchNorm1d(hidden_size * self.num_directions)
        self.dropout = nn.Dropout(config.drop_out)
        self.classifier = nn.Linear(hidden_size * self.num_directions, 1)
        self.sigmoid = nn.Sigmoid()
        # 将模型整体搬到指定 device
        self.to(device)
        self.config = config

    def forward(self, raw_input, GNN_output, mask):
        # GNN_output = GNN_output.transpose(0, 1)
        # GNN_output = GNN_output.float()
        # x = GNN_output.unsqueeze(-1)
        # x = GNN_output.transpose(0, 1).float()  # -> (batch, seq_len, input_size)
        x = GNN_output.float() 
        out, _ = self.lstm(x)      
        pooled = out.mean(dim=1) 
        # mean_output = self.bn(out)
        normalized = self.bn(pooled)            # (batch, hidden)
        dropped = self.dropout(normalized)

        logits = self.classifier(dropped) # (batch_size, 1)
        logits = self.sigmoid(logits).squeeze(1) # (batch_size)

        return logits, normalized