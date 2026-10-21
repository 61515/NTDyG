import sys

import torch
import torch.nn.functional as F
# from torch_geometric.nn import GCNConv
# from torch_geometric.data import Data
import torch.optim as optim
import torch.nn as nn

class CombinedModel(torch.nn.Module):
    # 这段 CombinedModel 的作用就是把一个图神经网络（GNN）和一个 Transformer 串起来，先让 GNN 处理变长的边／节点特征，然后把 GNN 的输出 pad 到统一长度，再交给 Transformer 做后续序列建模。
    def __init__(self, config, GNN, Transformer):
        super(CombinedModel, self).__init__()
        # self.input_size = config.input_dim
        self.num_classes = config.num_classes  # 包括“未知”在内的类别总数

        self.GNN = GNN
        self.Transformer = Transformer

    # def forward(self, H_e, H_e_pad, adj_e, adj_v, T, mask):
    def forward(self, H_v, H_e, H_e_pad, adj_e, adj_v, T, mask):
        # 通过第一个模型
        # 1. 用 GNN 处理每条图的可变长度输入
        # H_v = [h.t() for h in H_v]
        H_e = [h.t() for h in H_e]
        # H_e_pad = [h.t() for h in H_e_pad]

        output = self.GNN(H_v, H_e, adj_e, adj_v, T)
        #  假设 output 是一个长度 = batch_size 的列表，
        #  每个元素是形状 (L_i, D) 的张量（L_i 不同）。
        # 2. 手动 pad 到 (batch, max_L, D)
        pad_output = torch.zeros((H_e_pad.shape[0], H_e_pad.shape[2], H_e_pad.shape[1])).to(H_e[0].device)
        pad_He = torch.zeros((H_e_pad.shape[0], H_e_pad.shape[2], H_e_pad.shape[1])).to(H_e[0].device)
        for i in range(H_e_pad.shape[0]):
            length = output[i].shape[0]
            pad_output[i, :length, :] = output[i]
            L_i = H_e[i].shape[0]
            pad_He[i, :L_i, :] = H_e[i]
            

        # 通过第二个模型
        # 3. 把原始 pad（H_e_pad）和 GNN pad_output 一起送入 Transformer
        # output, feat1 = self.Transformer(pad_He, pad_output, mask)
        output, feat1 = self.Transformer(pad_He, pad_output, mask)


        return output, feat1

