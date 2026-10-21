import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .layers2 import GraphConvolution

class CensNet(nn.Module):
    # 这是一个图神经网络（GNN）模型，结合了两层自定义图卷积操作（GraphConvolution）并支持图的边和节点的联合表示学习，适用于一批图结构数据。
    def __init__(self, vfeat, nfeat, dropout):
        # dropout：dropout 概率，用于防止过拟合。
        super(CensNet, self).__init__()
        # self.gc1 = GraphConvolution(nfeat, nfeat, nfeat, nfeat, node_layer=True) # 节点？
        self.gc1 = GraphConvolution(vfeat, vfeat, nfeat, nfeat, node_layer=True) 
        # GraphConvolution(in_node_dim, in_edge_dim, out_node_dim, out_edge_dim, node_layer=False)
        self.gc2 = GraphConvolution(vfeat, vfeat, nfeat, nfeat, node_layer=False) # 边？
        # self.gc3 = GraphConvolution(nfeat, nfeat, nfeat, nfeat, node_layer=True) # 边？
        # self.gc4 = GraphConvolution(nfeat, nfeat, nfeat, nfeat, node_layer=False) # 边？
        self.dropout = dropout
        # self.gc1 = GraphConvolutionEdgeOnly(feat_dim, feat_dim)
        # self.gc2 = GraphConvolutionEdgeOnly(feat_dim, feat_dim)
        # self.dropout = dropout

    # def forward(self, X_list, Z_list, adj_e_list, adj_v_list, T_list):
    def forward(self, X_list, Z_list, adj_e_list, adj_v_list, T_list):
        # X_list: 每个图的 节点特征张量
        # Z_list: 每个图的 边特征张量
        # adj_e_list: 每个图的 边之间的邻接矩阵（edge adjacency matrix）
        # adj_v_list: 每个图的 节点之间的邻接矩阵（node adjacency matrix）
        # T_list: 每个图的 节点与边之间的连接张量（或交互矩阵）
        outputs = []
        for X, Z, adj_e, adj_v, T in zip(X_list, Z_list, adj_e_list, adj_v_list, T_list):
            # 转 float 并移动到同一 device
            X = X.float()
            Z = Z.float()
            adj_e = adj_e.float()
            adj_v = adj_v.float()
            T = T.float()
            device = Z.device

            # # 构造虚拟节点特征 X：全零，shape [num_nodes, nfeat]
            # num_nodes = adj_v.shape[0]
            # # 注意用与 Z 同 device
            # X = torch.ones((num_nodes, Z.shape[1]), device=device)
            gc1 = self.gc1(X, Z, adj_e, adj_v, T)
            X, Z = F.relu(gc1[0]), F.relu(gc1[1])
            
            X = F.dropout(X, self.dropout, training=self.training)
            # Z = F.dropout(Z, self.dropout, training=self.training)

            gc2 = self.gc2(X, Z, adj_e, adj_v, T)
            X, Z = F.relu(gc2[0]), F.relu(gc2[1])
            # X = F.dropout(X, self.dropout, training=self.training)
            Z = F.dropout(Z, self.dropout, training=self.training)
            
            # gc3 = self.gc3(X, Z, adj_e, adj_v, T)
            # X, Z = F.relu(gc3[0]), F.relu(gc3[1])
            # X = F.dropout(X, self.dropout, training=self.training)
            # Z = F.dropout(Z, self.dropout, training=self.training)
            
            # gc4 = self.gc4(X, Z, adj_e, adj_v, T)
            # X, Z = F.relu(gc4[0]), F.relu(gc4[1])
            # X = F.dropout(X, self.dropout, training=self.training)
            # Z = F.dropout(Z, self.dropout, training=self.training)

            outputs.append(Z)
            # 第一层 GraphConvolution：以边为主更新
            # X1, Z1 = self.gc1(X, Z, adj_e, adj_v, T)
            # 通常 node_layer=False 时 X1 可能与 X 无关或直接返回 X; 重点在 Z1
            # Z1 = F.relu(Z1)
            # Z1 = F.dropout(Z1, self.dropout, training=self.training)
            # 若需要，可以对 X1 也做激活/Dropout，但因无实际信息可选省略：
            # X1 = F.relu(X1); X1 = F.dropout(X1, self.dropout, training=self.training)

            # 第二层 GraphConvolution：继续以边为主更新
            # X2, Z2 = self.gc2(X1, Z1, adj_e, adj_v, T)
            # Z2 = F.relu(Z2)
            # Z2 = F.dropout(Z2, self.dropout, training=self.training)

            # outputs.append(Z2)
        return outputs

        # batch_size = len(T_list)
        # outputs = []
        # for i in range(batch_size): # 批量循环：对于 batch 中的每个样本（图）
        #     # 批量大小由 T_list 长度决定。
        #     # 每个图结构分开处理，适应不同大小（每个图节点数/边数可能不同）。
        #     X = X_list[i].float()
        #     Z = Z_list[i].float()
        #     adj_e = adj_e_list[i].float()
        #     adj_v = adj_v_list[i].float()
        #     T = T_list[i].float()

        #     gc1 = self.gc1(X, Z, adj_e, adj_v, T) # 调用第一层图卷积，返回一个 (X', Z') 对
        #     X, Z = F.relu(gc1[0]), F.relu(gc1[1])
        #     X = F.dropout(X, self.dropout, training=self.training)
        #     Z = F.dropout(Z, self.dropout, training=self.training)
            
        #     gc2 = self.gc2(X, Z, adj_e, adj_v, T)
        #     X, Z = F.relu(gc2[0]), F.relu(gc2[1])
        #     X = F.dropout(X, self.dropout, training=self.training)
        #     Z = F.dropout(Z, self.dropout, training=self.training)
        #     outputs.append(Z)
        # return outputs
    
# class GraphConvolutionEdgeOnly(nn.Module):
#     def __init__(self, in_edge_dim, out_edge_dim):
#         super(GraphConvolutionEdgeOnly, self).__init__()
#         self.edge_mlp = nn.Linear(in_edge_dim, out_edge_dim)

#     def forward(self, Z, adj_e, T):
#         # 假设：adj_e 是边与边之间的连接信息（类似边图）
#         Z_new = torch.matmul(adj_e, Z)  # 消息聚合（简化）
#         Z_new = self.edge_mlp(Z_new)    # 线性转换
#         return Z_new