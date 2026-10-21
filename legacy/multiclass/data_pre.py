import sys

import torch
import torch.utils.data
import os
import numpy as np
from option import args
import random
import pandas as pd
from operator import itemgetter
import pickle
import scipy.sparse as sp
from copy import deepcopy
from collections import defaultdict
import re
# 逐行检查，只保留所有列都满足为数值且不含中文的行
def check_row(row):
    for v in row:
        if not is_numeric(v):
            return False
        if has_chinese(v):
            return False
    return True

# 判断是否含有中文字符
def has_chinese(text):
    if isinstance(text, str):
        # 使用正则判断是否包含中文字符
        return bool(re.search(r'[\u4e00-\u9fff]', text))
    return False

def is_numeric(val):
    # 先判断是否为数字（整数或浮点数）
    if pd.isna(val):
        return False
    # 排除非字符串类型，尽量转换为float看看是否成功
    try:
        float(val)
        return True
    except (ValueError, TypeError):
        return False
    
class DygDataset(torch.utils.data.Dataset):
    # 用于图结构数据的加载、预处理和训练/测试划分
    def __init__(self, config, split_flag, train_path, test_path):
        self.config = config

        dataset_name = '{}/{}.pkl'.format(config.dir_data, config.data_set)

        with open(dataset_name, 'rb') as file: #读取
            data = pickle.load(file)
        # 调用 get_data() 方法对图节点、边、特征、标签等数据进一步构建。
        self.input_nodes_feature, self.input_edges_feature, self.input_edges_pad, self.labels2, self.Tmats, self.adjs, self.eadjs, self.mask_edge, self.idx_column = self.get_data(data, split_flag, train_path, test_path)

    def get_data(self, data, split_flag, train_path, test_path):
        node_features = data['nodefeatures']
        edge_features = data['edgefeatures']
        labels1 = data['label1s']
        labels2 = data['label2s']
        Tmats = data['Tmats']
        adjs = data['adjs']
        eadjs = data['eadjs']
        idx_column = data['Timestamp_Label'] 
        # flattened_node = np.concatenate([arr for arr in node_features]) # 将 node_features 中的多个 NumPy 数组按顺序拼接（concatenate）成一个更大的数组，并将结果赋给 flattened_node
        # unique_node = np.unique(flattened_node)
        # num_nodes = int(max(unique_node.size, np.max(unique_node) + 1))
        # Nfeatures = np.random.uniform(low=0.0, high=1.0, size=(num_nodes, 1))

        NS = len(edge_features) # 样本数（子图数）
        edge_features = [np.array(arr).reshape(-1, 1) if arr.ndim == 1 else np.array(arr)
                        for arr in edge_features]
        edge_feat_dims = [arr.shape[1] for arr in edge_features]
        # if len(set(edge_feat_dims)) != 1:
        #     raise ValueError(f"不同子图的边特征维度不一致: {set(edge_feat_dims)}")
        hidden_e = edge_feat_dims[0]
        # 2. 找到最长的 m_i（最大边数）
        # 生成边的 padding 掩码 (mask_edge)
        max_mask_edge = max(arr.shape[0] for arr in edge_features) # edge_features 是每个子图中“这条样本的边索引列表”，找出最长的那个长度
        # 3. 构造 mask_edge：shape (NS, max_mask_edge)，0 表示有效，1 表示 padding
        mask_edge = np.ones((NS, max_mask_edge), dtype=np.int64)
        # masks_edge = [len(edge) for edge in edge_features] # 每个子图实际边索引长度列表
        for i, arr in enumerate(edge_features):
            l = arr.shape[0]
            mask_edge[i, :l] = 0 # 将对应位置的 mask_edge[i, :l] 设置为 0,表示这些是 真实边的位置（有效）
        
        # 4. 构造 input_edges_pad: shape (NS, max_mask_edge, hidden_e)
        input_edges_pad = np.zeros((NS, max_mask_edge, hidden_e), dtype=np.float32)
        for i, arr in enumerate(edge_features):
            l = arr.shape[0]
            input_edges_pad[i, :l, :] = arr.astype(np.float32)

        input_edges_feature = [
            torch.tensor(edge_features[i], dtype=torch.float32)
            for i in range(NS)
        ]       
        

        NS = len(node_features)
        hidden = 1
        max_mask_node = max(len(arr) for arr in node_features)
        masks_node = [len(node) for node in node_features]
        input_nodes_pad = np.zeros((NS, max_mask_node, hidden))
        for i, indices in enumerate(node_features):
            indices = indices.astype(int)
            input_nodes_pad[i, :len(indices), :] = (1+indices.astype(np.float32)).reshape(-1, 1)

        input_nodes_feature = [torch.tensor(input_nodes_pad[i, :masks_node[i], :]) for i in range(NS)]
        # input_nodes_feature = [torch.tensor(node_features[i], dtype=torch.float32) for i in range(NS)]
  
        # 根据数据集类型划分训练集/测试集
        if self.config.data_set == 'Darknet':
            label_column2 =  labels2
            label_column2[labels1 == 0] = 0
            label_column2[labels1 == 1] = 0
            label_column2[labels1 == 2] = 1
            label_column2[labels1 == 3] = 1
            labels2 = label_column2


        elif self.config.data_set == 'CloudDDoS':
            label_column2 =  labels2
            label_column2[labels1 == 0] = 0
            label_column2[labels1 == 1] = 1
            labels2 = label_column2
        
  
        elif self.config.data_set == 'IIoTAPT':
            indices1 = np.where((labels1 == 0))[0]
            indices2 = np.where((labels1 == 1))[0]

        if split_flag == 'train':
            dataset_name_indx1 = '{}/{}_finetune_indx.csv'.format(self.config.dir_data, self.config.data_set)
            # df_indx = pd.DataFrame(indices2)
            # df_indx.to_csv(dataset_name_indx, index=False, header=None, encoding="utf-8")
            indices1 = pd.read_csv(dataset_name_indx1, header=None, encoding="utf-8").iloc[:, 0].to_numpy()
        
            input_nodes_feature = [input_nodes_feature[i] for i in indices1]
            input_edges_feature = [input_edges_feature[i] for i in indices1]
            input_edges_pad = [input_edges_pad[i] for i in indices1]
            # input_edges_pad = input_edges_pad[indices1]
            labels2 = [labels2[i] for i in indices1]
            Tmats = [Tmats[i] for i in indices1]
            adjs = [adjs[i] for i in indices1]
            eadjs = [eadjs[i] for i in indices1]
            mask_edge = [mask_edge[i] for i in indices1]
            idx_column = [idx_column[i] for i in indices1]
            idx_column = list(range(len(idx_column)))
            
            input_edges_pad = torch.tensor(input_edges_pad)
            labels2 = torch.tensor(labels2)
            mask_edge = torch.tensor(mask_edge)
            idx_column = torch.tensor(idx_column)

            # dataset_train = input_nodes_feature, input_edges_feature, input_edges_pad, labels2, Tmats, adjs, eadjs, mask_edge
            # torch.save(dataset_train, train_path, _use_new_zipfile_serialization=False)

        elif split_flag == 'test':
            dataset_name_indx2 = '{}/{}_test_indx.csv'.format(self.config.dir_data, self.config.data_set, self.config.datatype)
            # df_indx = pd.DataFrame(indices2)
            # df_indx.to_csv(dataset_name_indx, index=False, header=None, encoding="utf-8")
            indices2 = pd.read_csv(dataset_name_indx2, header=None, encoding="utf-8").iloc[:, 0].to_numpy()

            input_nodes_feature = [input_nodes_feature[i] for i in indices2]
            input_edges_feature = [input_edges_feature[i] for i in indices2]
            input_edges_pad = [input_edges_pad[i] for i in indices2]
            labels2 = [labels2[i] for i in indices2]
            Tmats = [Tmats[i] for i in indices2]
            adjs = [adjs[i] for i in indices2]
            eadjs = [eadjs[i] for i in indices2]
            mask_edge = [mask_edge[i] for i in indices2]
            idx_column = [idx_column[i] for i in indices2]
            idx_column = list(range(len(idx_column)))

            input_edges_pad = torch.tensor(input_edges_pad)
            labels2 = torch.tensor(labels2)
            mask_edge = torch.tensor(mask_edge)
            idx_column = torch.tensor(idx_column)
            # dataset_test = input_nodes_feature, input_edges_feature, input_edges_pad, labels2, Tmats, adjs, eadjs, mask_edge

            # torch.save(dataset_test, test_path, _use_new_zipfile_serialization=False)
        
        # return dataset_train, dataset_test
        return input_nodes_feature, input_edges_feature, input_edges_pad, labels2, Tmats, adjs, eadjs, mask_edge, idx_column

    def __getitem__(self, item):
        # 返回一个样本的所有相关信息（边特征、节点特征、邻接矩阵、标签等）作为一个字典。
        sinput_nodes_feature = self.input_nodes_feature[item]
        sinput_edges_feature = self.input_edges_feature[item]
        sinput_edges_pad = self.input_edges_pad[item]
        slabels2 = self.labels2[item]
        sTmats = self.Tmats[item]
        sadjs = self.adjs[item]
        seadjs = self.eadjs[item]
        smask_edge = self.mask_edge[item]
        sidx_column = self.idx_column[item]

        # sample = {
        #     'input_nodes_feature': sinput_nodes_feature,
        #     'input_edges_feature': sinput_edges_feature,
        #     'input_edges_pad': sinput_edges_pad,
        #     'labels2': slabels2,
        #     'Tmats': sTmats,
        #     'adjs': sadjs,
        #     'eadjs': seadjs,
        #     'mask_edge': smask_edge
        # }

        return {
            'input_nodes_feature': sinput_nodes_feature,
            'input_edges_feature': sinput_edges_feature,
            'input_edges_pad': sinput_edges_pad,
            'labels2': slabels2,
            'Tmats': sTmats,
            'adjs': sadjs,
            'eadjs': seadjs,
            'mask_edge':smask_edge,
            'Timestamp_Label':sidx_column
        }

    def __len__(self):
        return len(self.labels2)


class Collate: # 这个 Collate 类封装了一个自定义的 “拼 batch” 函数（collate function），通常配合 DataLoader 使用，用来把若干个样本整合成一个 batch。
    def __init__(self, config):
        self.config = config
    # Collate 是一个自定义的 批处理封装类，通常用于 torch.utils.data.DataLoader 中的 collate_fn 参数，用来把若干个单独样本组合成一个 batch。
    # 尤其在图神经网络（如动态图）场景中，每个样本的结构和维度可能不一致（如邻接矩阵、边数量、节点数量不同），所以需要特别定制。
    def dyg_collate_fn(self, batch): # 这个函数接受一个 batch（列表形式，每个元素是一个样本，是 dict 类型），将它们聚合成统一格式。
        input_nodes_feature = [b['input_nodes_feature'] for b in batch] # 保留为列表结构，适应每个样本边数不一致的情况。
        input_edges_feature = [b['input_edges_feature'] for b in batch] # 这行代码的作用是将当前 batch 中每个样本的 input_edges_feature 字段提取出来，聚合成一个 Python 列表，方便后续模型按样本访问或进一步处理
        input_edges_pad = torch.stack([b['input_edges_pad'] for b in batch], dim=0)
        # 这一行代码的作用是把每个样本中已经 pad 好并且形状相同的 input_edges_pad 张量，沿着“新”加的第 0 维，也就是 batch 维，拼成一个整体的张量。
        labels2 = torch.stack([b['labels2'] for b in batch], dim=0)

        Tmats = [b['Tmats'] for b in batch]
        adjs = [b['adjs'] for b in batch]
        eadjs = [b['eadjs'] for b in batch]
        mask_edge = torch.stack([b['mask_edge'] for b in batch], dim=0)

        idx_column = torch.stack([b['Timestamp_Label'] for b in batch], dim=0)

        return {
            'input_nodes_feature': input_nodes_feature,
            'input_edges_feature': input_edges_feature,
            'input_edges_pad': input_edges_pad,
            'labels2': labels2,
            'Tmats': Tmats,
            'adjs': adjs,
            'eadjs': eadjs,
            'mask_edge':mask_edge,
            'Timestamp_Label':idx_column
        }




if __name__ == '__main__':
    config = args
    a = DygDataset(config, 'train')
    # a = DygDatasetTest(config, 'val')
    c = a[5000]
    print(c)
