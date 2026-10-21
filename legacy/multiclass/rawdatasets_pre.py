import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["LOKY_MAX_CPU_COUNT"] = "8"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from tqdm import tqdm
import pickle
from option import args
import pandas as pd
import torch
import numpy as np
import scipy.sparse as sp
from scipy.spatial.distance import pdist, squareform
import sys
import random
import copy
import matplotlib.pyplot as plt
import re
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool, cpu_count
import concurrent.futures
from sklearn.cluster import KMeans
import networkx as nx
from sklearn.preprocessing import MinMaxScaler



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

# 判断是否含有中文字符
def has_chinese(text):
    if isinstance(text, str):
        # 使用正则判断是否包含中文字符
        return bool(re.search(r'[\u4e00-\u9fff]', text))
    return False

# 逐行检查，只保留所有列都满足为数值且不含中文的行
def check_row(row):
    for v in row:
        if not is_numeric(v):
            return False
        if has_chinese(v):
            return False
    return True


class BatchGraphSample:
    def __init__(self, config, graph_df):
        self.config = config
        self.graph_df = graph_df
        # 生成图
        # self.G = nx.Graph() # 创建一个空的无向图
        self.G = nx.DiGraph()
        self.max_mask_len = 100
        self.k = 1
        for index, row in graph_df.iterrows():
            #遍历 graph_df 的每一行，把 (u, v) 作为图中的一条无向边。
            # 如果同一对 (u,v) 在 CSV 里出现多次，就会把多个 id（或“weight”）都保存到同一条边的属性 weight 中
            # （变成一个 Python 列表）。
            u, v = row['Src IP Label'], row['Dst IP Label']
            weight = row.drop(['Src IP Label', 'Dst IP Label', 'Timestamp_Label', 'Label2_encoded', 'Label1_encoded'])
            IDTime = row['Timestamp_Label']
            
            if self.G.has_edge(u, v):
                attr = self.G[u][v]
                if isinstance(attr['weight'], list):
                    attr['weight'].append(weight)
                else:
                # 如果之前存的是单值，转换为列表
                    attr['weight'] = [attr['weight'], weight]
                
                if isinstance(attr['IDTime'], list):
                    attr['IDTime'].append(IDTime)
                else:
                # 如果之前存的是单值，转换为列表
                    attr['IDTime'] = [attr['IDTime'], IDTime]

                # 如果边已经存在，将新权重添加到现有权重列表
                # self.G[u][v]['weight'].append(weight)
                # self.G[u][v]['IDTime'].append(IDTime)
            else:
                # 如果边不存在，创建新的边并初始化权重列表
                self.G.add_edge(u, v, weight=[weight], IDTime=[IDTime])
                # self.G.add_edge(u, v, IDTime=[IDTime])

        
    # 替换子图中的节点和边
    def replace_subgraph(self, subgraph, node_mapping, src, dst):

        new_subgraph = nx.Graph()
        # feature_dim = subgraph[src][dst].get('weight', None)
        num_nodes = len(subgraph.nodes())
        # new_node_features = np.zeros(num_nodes)
        # new_node_features = np.ones((num_nodes, feature_dim.shape[0])) 
        # new_node_features = np.ones((num_nodes, 1)) 
        new_node_features = np.zeros(num_nodes)

        for node, new_node_id in node_mapping.items():
            new_node_features[new_node_id] = int(node)
            new_subgraph.add_node(new_node_id)
        # new_node_features[node_mapping[src]] = subgraph[src][dst]['weight'].map(int)
        # new_node_features[node_mapping[dst]] = subgraph[src][dst]['weight'].map(int)

        new_subgraph.add_edge(node_mapping[src], node_mapping[dst])
        
        mask = new_node_features.shape[0] # 表示节点数

        return new_subgraph, new_node_features, mask

    def create_transition_matrix(self, vertex_adj):
        '''create N_v * N_e transition matrix'''
        # 构造一个从 节点到边的转移矩阵T
        # 节点数量，边的数量（无向图按一次计数），T[i, j] = 1 表示 第 i 个节点属于第 j 条边
        vertex_adj.setdiag(0)
        # 将邻接矩阵对角线置零，去掉自连接（避免将自环当成一条普通边处理）
        edge_index = np.nonzero(sp.triu(vertex_adj, k=1))
        # 使用 sp.triu(..., k=1) 获取邻接矩阵的上三角部分（不包括对角线）
        # 这样做是因为无向图中 (i, j) 和 (j, i) 是一条边，只保留上三角防止重复。
        # edge_index 是一个 (row, col) 索引元组，表示存在边的位置。
        num_edge = int(len(edge_index[0]))
        edge_name = [x for x in zip(edge_index[0], edge_index[1])]
        # 得到总边数 num_edge 和每条边的 (u, v) 组合
        row_index = [i for sub in edge_name for i in sub]
        col_index = np.repeat([i for i in range(num_edge)], 2)
        # 构造坐标对用于稀疏矩阵构建：row_index 是所有节点编号（每条边两个点）
        # col_index 是边编号，每个边号重复两次，对应两个端点
        data = np.ones(num_edge * 2)
        T = sp.csr_matrix((data, (row_index, col_index)),
                          shape=(vertex_adj.shape[0], num_edge))
        # 其中每列表示一条边，1 表示该节点属于该边。
        return T

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        # tocoo()：转换为坐标格式（COO），方便提取 (row, col, value)
        # 确保所有数据是 float32 类型，符合 torch.FloatTensor 要求
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        # sparse_mx.row, sparse_mx.col：两个数组，表示非零元素的坐标
        # vstack(...).astype(np.int64)：生成形状为 [2, N] 的索引张量，适用于 PyTorch 稀疏张量
        values = torch.from_numpy(sparse_mx.data)
        # sparse_mx.data 是对应坐标下的值数组（float32）
        shape = torch.Size(sparse_mx.shape)
        # return torch.sparse.FloatTensor(indices, values, shape)
        return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)
    # 构建 PyTorch 稀疏张量：稀疏索引+非零值+形状

    def create_edge_adj(self, vertex_adj):
        # 为了根据顶点邻接矩阵 vertex_adj，构造一个 边邻接矩阵 edge_adj。它的设计用于建立 边-边图（line graph），其中：
        # 每条边变成一个“节点”，
        # 若两个边共享同一个顶点，则它们在边图中是相邻的。
        '''
        create an edge adjacency matrix from vertex adjacency matrix
        '''
        vertex_adj.setdiag(0)
        # 移除对角线（去掉自环）
        edge_index = np.nonzero(sp.triu(vertex_adj, k=1))
        num_edge = int(len(edge_index[0]))
        edge_name = [x for x in zip(edge_index[0], edge_index[1])]
        # 找出所有唯一边 (i, j)（无向图只保留一半）
        # edge_name[i] = (u, v) 表示第 i 条边连接的两个节点

        # 构造边邻接矩阵
        edge_adj = np.zeros((num_edge, num_edge))
        for i in range(num_edge):
            for j in range(i, num_edge):
                if len(set(edge_name[i]) & set(edge_name[j])) == 0:
                    edge_adj[i, j] = 0
                else:
                    edge_adj[i, j] = 1
        # 遍历所有边对 (i, j)
        # 如果两条边有共同端点（set(edge_name[i]) & set(edge_name[j]) 非空），它们在边图中相连
        adj = edge_adj + edge_adj.T
        np.fill_diagonal(adj, 1)
        # 保证最终是对称矩阵（无向边图）
        # 对角线填 1，表示自己与自己相连
        # 返回稀疏格式和边名（方便后续使用）
        return sp.csr_matrix(adj), edge_name

        # 假设 new_subgraph 是一个 networkx.Graph 对象

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1)).astype("float")
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    
    def subgraph_edges_to_src(self, G, nodes, src_node, dst_label):
        """
        从图 G 中，针对给定节点列表 nodes 和 src_node，构造一个子图：
        只保留所有存在于 G 中的 (node, src_node) 边及其端点节点。
        """
        # 收集所有在 G 中确实与 src_node 相连的边
        edges = []
        for node in nodes:
            # 对无向图，has_edge(node, src_node) 就够了；对有向图可根据需要检查方向
            if G.has_edge(node, src_node):
                edges.append((node, src_node))
            # elif G.is_directed() and G.has_edge(src_node, node):
            #     # 若想在有向图中也保留反向边，可加入这行
            #     edges.append((src_node, node))
        # 使用 edge_subgraph 构造子图（视图），然后拷贝为独立图
        edges.append((src_node, dst_label))
        sub = G.edge_subgraph(edges).copy()
        return sub


    def _process_edge(self, edge_row):
        """
        对单条 edge（Series 或类似 dict-like）做和原来循环体里一致的处理逻辑，返回：
        new_node_features, edge_feature_matrix, tensor_T, adj_tensor, eadj_tensor, mask
        """
        src_label = edge_row['Src IP Label']
        dst_label = edge_row['Dst IP Label']
        timestamp_label = edge_row['Timestamp_Label']
        weight_label = edge_row.drop(['Src IP Label', 'Dst IP Label', 'Timestamp_Label', 'Label2_encoded', 'Label1_encoded'])

        nodes = {src_label, dst_label}
        # 2. 从全局图提取子图，并拷贝一份用于局部修改
        # batch_subgraph = self.G.subgraph(nodes).copy()
        batch_subgraph = self.subgraph_edges_to_src(self.G, nodes, src_label, dst_label)

        # 4. 强制将当前 edge 的 参数与原来一样（当前需要算的边）
        if batch_subgraph.has_edge(src_label, dst_label):
            batch_subgraph[src_label][dst_label]['IDTime'] = timestamp_label
            batch_subgraph[src_label][dst_label]['weight'] = weight_label

        else:
            print(f"[Warning] Error processing edge ({src_label}, {dst_label})")
            sys.exit()

        
        # 5. reorder_nodes, replace_subgraph 重新排序节点，替换子图
        node_mapping = {src_label: 0, dst_label: 1}
        new_subgraph, new_node_features, mask = self.replace_subgraph(batch_subgraph, node_mapping, src_label, dst_label)
        
        # 6. 构建邻接矩阵、转 torch 稀疏、transition matrix
        adj = nx.adjacency_matrix(new_subgraph)  # SciPy sparse 矩阵
        T = self.create_transition_matrix(adj)   # 返回 SciPy sparse 矩阵?
        tensor_T = self.sparse_mx_to_torch_sparse_tensor(T)


        edge_feature_matrix = batch_subgraph[src_label][dst_label]['weight'].values
   

        # 9. 归一化并转 torch sparse
        # eadj = self.sparse_mx_to_torch_sparse_tensor(eadj)
        # adj = self.sparse_mx_to_torch_sparse_tensor(adj + sp.eye(adj.shape[0]))
        eadj = self.sparse_mx_to_torch_sparse_tensor(self.normalize(sp.coo_matrix([[1.0]])))
        adj = self.sparse_mx_to_torch_sparse_tensor(self.normalize(adj + sp.eye(adj.shape[0])))

        # 10. 返回
        return new_node_features, edge_feature_matrix, tensor_T, adj, eadj, mask
    
    def get_batch_data(self, idx_batch, num_workers=None):
        """
        并行处理 idx_batch 对应的多条边，返回与原函数相同格式的 tuple of lists。
        idx_batch: 可迭代，内部元素与 graph_df['Timestamp_Label'] 对应
        num_workers: 线程数，默认 None 则使用 os.cpu_count() 或 1。
        """
        # 过滤出需要处理的 edges DataFrame
        edges = self.graph_df[self.graph_df['Timestamp_Label'].isin(idx_batch)]
        # edges = edges.set_index('Timestamp_Label').loc[idx_batch].reset_index() # 保证顺序一致
        # 确定并行线程数
        if num_workers is None:
            num_workers = min(32, (os.cpu_count() or 1) + 4)  # 可根据机器调整
        # 用 ThreadPoolExecutor 避免多进程序列化开销
        results = []
        # # 使用 tqdm 显示进度
        # # with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        # with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:    
        #     # 将每行 edge_row 传给 _process_edge
        #     futures = []
        #     for _, edge_row in edges.iterrows():
        #         # iterrows() 是 Pandas 提供的行迭代器，返回 (index, Series)。
        #         # 注意：pandas Series 是可在闭包中引用的，但读取时会被复制为小对象
        #         futures.append(executor.submit(self._process_edge, edge_row))
        #         # executor 是一个线程池（来自 ThreadPoolExecutor 或 ProcessPoolExecutor）。
        #         # submit(...) 会把任务提交给线程池异步运行，立即返回一个 Future 对象。
        #         # self._process_edge(edge_row) 是你定义的单条边的处理逻辑，每条边并行处理。
        #     # as_completed + tqdm
        #     for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing edges"):
        #         # results.append(f.result())
        #         # try:
        #         results.append(f.result())
        #         # except Exception as e:
        #         #     print(f"某个任务出错: {e}")

        for _, edge_row in tqdm(edges.iterrows(), total=len(edges), desc="Processing edges"):
            # try:
            result = self._process_edge(edge_row)
            results.append(result)
            # except Exception as e:
            #     print(f"某个任务出错: {e}")

        all_node_features, all_edge_features, all_Tmat, all_adj, all_eadj, all_mask = zip(*results)
        # 转成 list 返回
        return list(all_node_features), list(all_edge_features), list(all_Tmat), list(all_adj), list(all_eadj), list(all_mask)
    

if __name__ == '__main__':
  
    config = args
    # for config.data_set in ["CloudDDoS"]:
    # for config.data_set in ["IIoTAPT"]:
    for config.data_set in ["Darknet"]:
        if config.data_set == 'Darknet':
            dataset_name = '{}/{}'.format(config.dir_data, config.data_set)
            file_path = os.path.join(config.dir_data, f"{config.data_set}_re.csv")

            if os.path.isfile(file_path):
                graph_df = pd.read_csv('{}_re.csv'.format(dataset_name))
                #归一化
                # 2. 提取三列，转成 NumPy（或保留为 Series 也可以）
                label_column1 = graph_df['Label1_encoded'].to_numpy()
                label_column2 = graph_df['Label2_encoded'].to_numpy()

                label_column2[label_column1 == 0] = 0
                label_column2[label_column1 == 1] = 0
                label_column2[label_column1 == 2] = 1
                label_column2[label_column1 == 3] = 1

                idx_column    = graph_df['Timestamp_Label'].to_numpy()

                src_column    = graph_df['Src IP Label'].to_numpy()
                dst_column    = graph_df['Dst IP Label'].to_numpy()
                # 3. 挑出“其他所有列”
                cols_to_exclude = ['Label1_encoded', 'Label2_encoded', 'Timestamp_Label', 'Src IP Label', 'Dst IP Label']
                feature_cols = [c for c in graph_df.columns if c not in cols_to_exclude]

                #   4. 最小‑最大归一化
                scaler = MinMaxScaler(feature_range=(0,1))
                # fit_transform 会返回一个 (n_samples, n_features) 的 np.ndarray
                features_norm = scaler.fit_transform(graph_df[feature_cols])

                # 5. 把归一化结果打包回 DataFrame（保留原索引和列名）
                df_norm = pd.DataFrame(features_norm,
                       index=graph_df.index,
                       columns=feature_cols)

                # 6. 再把三列拼回去
                df_norm['Label1_encoded']    = label_column1
                df_norm['Label2_encoded']    = label_column2
                df_norm['Timestamp_Label']   = idx_column
                df_norm['Src IP Label']   = src_column
                df_norm['Dst IP Label']   = dst_column
                graph_df = df_norm
           
            graph_df = graph_df[graph_df.apply(check_row, axis=1)].reset_index(drop=True)
            label_column1 = graph_df['Label1_encoded'].to_numpy() # 把所有样本的 label 列先保存下来；
            label_column2 = graph_df['Label2_encoded'].to_numpy() # 把所有样本的 label 列先保存下来；
            idx_column = graph_df['Timestamp_Label'].to_numpy() # 把每一行的 id（即边的唯一标识或“权重”）取出来，形成一个一维 NumPy 数组，后面作为“要采样的边 ID 清单”；


            a = BatchGraphSample(config, graph_df) # 构造一个 “批次子图采样器” 对象 a
            all_node_features, all_edge_features, all_Tmat, all_adj, all_eadj, all_mask = a.get_batch_data(
        idx_column) # （遍历采集子图） 给它一串要处理（采样）边的 ID 列表，它会对每个边 ID 都提取一个 1-hop（k=1）子图，并做一系列的矩阵／向量处理，最后返回 6 个列表
        
        if config.data_set == 'CloudDDoS':
            
            dataset_name = '{}/{}'.format(config.dir_data, config.data_set)
            file_path = os.path.join(config.dir_data, f"{config.data_set}_re.csv")

            if os.path.isfile(file_path):
                graph_df = pd.read_csv('{}_re.csv'.format(dataset_name))
                #归一化
                # 2. 提取三列，转成 NumPy（或保留为 Series 也可以）
                label_column1 = graph_df['Label1_encoded'].to_numpy()
                label_column2 = graph_df['Label2_encoded'].to_numpy()

                label_column2[label_column1 == 0] = 0
                label_column2[label_column1 == 1] = 1

                idx_column    = graph_df['Timestamp_Label'].to_numpy()

                src_column    = graph_df['Src IP Label'].to_numpy()
                dst_column    = graph_df['Dst IP Label'].to_numpy()
                # 3. 挑出“其他所有列”
                cols_to_exclude = ['Label1_encoded', 'Label2_encoded', 'Timestamp_Label', 'Src IP Label', 'Dst IP Label']
                feature_cols = [c for c in graph_df.columns if c not in cols_to_exclude]

                #   4. 最小‑最大归一化
                scaler = MinMaxScaler(feature_range=(0,1))
                # fit_transform 会返回一个 (n_samples, n_features) 的 np.ndarray
                features_norm = scaler.fit_transform(graph_df[feature_cols])

                # 5. 把归一化结果打包回 DataFrame（保留原索引和列名）
                df_norm = pd.DataFrame(features_norm,
                       index=graph_df.index,
                       columns=feature_cols)

                # 6. 再把三列拼回去
                df_norm['Label1_encoded']    = label_column1
                df_norm['Label2_encoded']    = label_column2
                df_norm['Timestamp_Label']   = idx_column
                df_norm['Src IP Label']   = src_column
                df_norm['Dst IP Label']   = dst_column
                graph_df = df_norm
           
            graph_df = graph_df[graph_df.apply(check_row, axis=1)].reset_index(drop=True)
            label_column1 = graph_df['Label1_encoded'].to_numpy() # 把所有样本的 label 列先保存下来；
            label_column2 = graph_df['Label2_encoded'].to_numpy() # 把所有样本的 label 列先保存下来；
            idx_column = graph_df['Timestamp_Label'].to_numpy() # 把每一行的 id（即边的唯一标识或“权重”）取出来，形成一个一维 NumPy 数组，后面作为“要采样的边 ID 清单”；
            

            a = BatchGraphSample(config, graph_df) # 构造一个 “批次子图采样器” 对象 a
            all_node_features, all_edge_features, all_Tmat, all_adj, all_eadj, all_mask = a.get_batch_data(
        idx_column) # （遍历采集子图） 给它一串要处理（采样）边的 ID 列表，它会对每个边 I
        
        if config.data_set == 'IIoTAPT':
            dataset_name = '{}/{}'.format(config.dir_data, config.data_set)
            file_path = os.path.join(config.dir_data, f"{config.data_set}_re.csv")

            if os.path.isfile(file_path):
                graph_df = pd.read_csv('{}_re.csv'.format(dataset_name))

                
                #归一化
                # 2. 提取三列，转成 NumPy（或保留为 Series 也可以）
                label_column1 = graph_df['Label1_encoded'].to_numpy()
                label_column2 = graph_df['Label2_encoded'].to_numpy()

                idx_column    = graph_df['Timestamp_Label'].to_numpy()

                src_column    = graph_df['Src IP Label'].to_numpy()
                dst_column    = graph_df['Dst IP Label'].to_numpy()
                # 3. 挑出“其他所有列”
                cols_to_exclude = ['Label1_encoded', 'Label2_encoded', 'Timestamp_Label', 'Src IP Label', 'Dst IP Label']
                feature_cols = [c for c in graph_df.columns if c not in cols_to_exclude]

                #   4. 最小‑最大归一化
                scaler = MinMaxScaler(feature_range=(0,1))
                # fit_transform 会返回一个 (n_samples, n_features) 的 np.ndarray
                features_norm = scaler.fit_transform(graph_df[feature_cols])

                # 5. 把归一化结果打包回 DataFrame（保留原索引和列名）
                df_norm = pd.DataFrame(features_norm,
                       index=graph_df.index,
                       columns=feature_cols)

                # 6. 再把三列拼回去
                df_norm['Label1_encoded']    = label_column1
                df_norm['Label2_encoded']    = label_column2
                df_norm['Timestamp_Label']   = idx_column
                df_norm['Src IP Label']   = src_column
                df_norm['Dst IP Label']   = dst_column
                graph_df = df_norm
           
            graph_df = graph_df[graph_df.apply(check_row, axis=1)].reset_index(drop=True)
            label_column1 = graph_df['Label1_encoded'].to_numpy() # 把所有样本的 label 列先保存下来；
            label_column2 = graph_df['Label2_encoded'].to_numpy() # 把所有样本的 label 列先保存下来；
            idx_column = graph_df['Timestamp_Label'].to_numpy() # 把每一行的 id（即边的唯一标识或“权重”）取出来，形成一个一维 NumPy 数组，后面作为“要采样的边 ID 清单”；
            

            a = BatchGraphSample(config, graph_df) # 构造一个 “批次子图采样器” 对象 a
            all_node_features, all_edge_features, all_Tmat, all_adj, all_eadj, all_mask = a.get_batch_data(
        idx_column) # （遍历采集子图） 给它一串要处理（采样）边的 ID 列表，它会对每个边 I
            
        # else:
        #     print("error!")

    all_mask = np.array(all_mask)
    data = {
        'nodefeatures': all_node_features,
        'edgefeatures': all_edge_features,
        'masks': all_mask,
        'label2s': label_column2,
        'label1s': label_column1,
        'Tmats': all_Tmat,
        'adjs': all_adj,
        'eadjs': all_eadj,
        'Timestamp_Label': idx_column
    }
    with open('{}/{}'.format(config.dir_data, config.data_set) + '.pkl', 'wb') as f: # 你这行代码的作用是将一个 Python 对象 data 序列化保存到本地磁盘，生成一个以 .pkl 结尾的二进制文件
        pickle.dump(data, f)
    
    print('--Data OK--')
