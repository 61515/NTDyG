import torch
import torch.nn as nn
from option import args
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import numpy as np
import torch.nn.functional as F


class BaseModel(nn.Module):
    def __init__(self, dataloader):
        super(BaseModel, self).__init__()
        self.num_users = dataloader.num_users
        self.num_items = dataloader.num_items
        self.emb_size = args.emb_size

    def forward(self):
        pass

    def cal_loss(self, batch_data):
        pass

    def _check_inf(self, loss, pos_score, neg_score, edge_weight):
        # find inf idx
        # 检查 loss 中是否出现 inf 或 NaN，若有则打印相关信息（pos/neg 分数与 edge_weight 的异常位置）并抛出异常，便于定位数值问题。
        inf_idx = torch.isinf(loss) | torch.isnan(loss)
        if inf_idx.any():
            print("find inf in loss")
            if type(edge_weight) != int:
                print(edge_weight[inf_idx])
            print(f"pos_score: {pos_score[inf_idx]}")
            print(f"neg_score: {neg_score[inf_idx]}")
            raise ValueError("find inf in loss")

    def _make_binorm_adj(self, mat, device):
        # 把用户×物品二分图转换成对称的全局邻接矩阵并做 D^-0.5 A D^-0.5 归一化，最后返回 PyTorch 稀疏张量（COO）
        a = csr_matrix((self.num_users, self.num_users))
        b = csr_matrix((self.num_items, self.num_items))
        mat = sp.vstack(
            [sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
        mat = (mat != 0) * 1.0
        # mat = (mat + sp.eye(mat.shape[0])) * 1.0# MARK
        degree = np.array(mat.sum(axis=-1))
        #
        degree = degree.astype(np.float64)
        d_inv_sqrt = np.zeros_like(degree, dtype=np.float64)
        nonzero = degree > 0
        d_inv_sqrt[nonzero] = np.power(degree[nonzero], -0.5)
        d_inv_sqrt = d_inv_sqrt.reshape(-1)
        # d_inv_sqrt = np.reshape(np.power(degree, -0.5), [-1])
        #
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_inv_sqrt_mat = sp.diags(d_inv_sqrt)
        mat = mat.dot(d_inv_sqrt_mat).transpose().dot(
            d_inv_sqrt_mat).tocoo()

        # make torch tensor
        idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = torch.from_numpy(mat.data.astype(np.float32))
        shape = torch.Size(mat.shape)

        # return torch.sparse.FloatTensor(idxs, vals, shape).to(device)
        sparse_tensor = torch.sparse_coo_tensor(indices=idxs, values=vals, size=shape, dtype=torch.float32, device=device)
        sparse_tensor = sparse_tensor.coalesce()
        return sparse_tensor
    
    def _make_binorm_adj_self_loop(self, mat):
        # 带自环版本
        a = csr_matrix((self.num_users, self.num_users))
        b = csr_matrix((self.num_items, self.num_items))
        mat = sp.vstack(
            [sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
        mat = (mat != 0) * 1.0
        mat = (mat + sp.eye(mat.shape[0])) * 1.0 # self loop
        degree = np.array(mat.sum(axis=-1))
        d_inv_sqrt = np.reshape(np.power(degree, -0.5), [-1])
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_inv_sqrt_mat = sp.diags(d_inv_sqrt)
        mat = mat.dot(d_inv_sqrt_mat).transpose().dot(
            d_inv_sqrt_mat).tocoo()

        # make torch tensor
        idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = torch.from_numpy(mat.data.astype(np.float32))
        shape = torch.Size(mat.shape)
        return torch.sparse.FloatTensor(idxs, vals, shape).to(args.device)


    def _sp_matrix_to_sp_tensor(self, sp_matrix):
        # 把任意 SciPy 稀疏矩阵转为 PyTorch 稀疏张量并做 .coalesce()（合并重复索引，按需要排序）。
        coo = sp_matrix.tocoo()
        indices = torch.LongTensor([coo.row, coo.col])
        values = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(indices, values, coo.shape).coalesce().to(args.device)

    def _bpr_loss(self, user_emb, pos_item_emb, neg_item_emb):
        pos_score = (user_emb * pos_item_emb).sum(dim=1)
        neg_score = (user_emb * neg_item_emb).sum(dim=1)
        loss = -torch.log(1e-10 + torch.sigmoid((pos_score - neg_score)))
        self._check_inf(loss, pos_score, neg_score, 0)
        return loss.mean()
    # def _reg_loss(self, users, pos_items, neg_items):
    #     u_emb = self.user_embedding[users]
    #     pos_i_emb = self.item_embedding[pos_items]
    #     neg_i_emb = self.item_embedding[neg_items]
    #     reg_loss = (1/2)*(u_emb.norm(2).pow(2) +
    #                       pos_i_emb.norm(2).pow(2) +
    #                       neg_i_emb.norm(2).pow(2))/float(len(users))
    #     return reg_loss
    def _nce_loss(self, pos_score, neg_score, edge_weight=1):
        numerator = torch.exp(pos_score)
        denominator = torch.exp(pos_score) + torch.exp(neg_score).sum(dim=1)
        loss = -torch.log(numerator/denominator) * edge_weight
        self._check_inf(loss, pos_score, neg_score, edge_weight)
        return loss.mean()
    
    def _infonce_loss(self, pos_1, pos_2, negs, tau):
        pos_1 = self.cl_mlp(pos_1)
        pos_2 = self.cl_mlp(pos_2)
        negs = self.cl_mlp(negs)
        pos_1 = F.normalize(pos_1, dim=-1)
        pos_2 = F.normalize(pos_2, dim=-1)
        negs = F.normalize(negs, dim=-1)
        pos_score = torch.mul(pos_1, pos_2).sum(dim=1)
        # B, 1, E * B, E, N -> B, N
        neg_score = torch.bmm(pos_1.unsqueeze(1), negs.transpose(1, 2)).squeeze(1)
        # infonce loss
        numerator = torch.exp(pos_score / tau)
        denominator = torch.exp(pos_score / tau) + torch.exp(neg_score / tau).sum(dim=1)
        loss = -torch.log(numerator/denominator)
        self._check_inf(loss, pos_score, neg_score, 0)
        return loss.mean()
    