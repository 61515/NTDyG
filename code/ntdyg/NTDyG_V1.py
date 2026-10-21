import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.nn.functional as F
from option import args
from utils import setup_logger, EarlyStopMonitor, set_seed
from utils import bpr_loss_binary_normal_only, eval_epoch_train_finetune, eval_epoch_finetune_one
import data_pre as dataset

from model.CensNet import CensNet
from model.Combine import CombinedModel

from model.Transformer import TransformerTargetEdgeClassifier
import numpy as np
from sklearn.ensemble import IsolationForest

from tqdm import tqdm
def main():
    for args.data_set in ["IIoTAPT"]:
    # for args.data_set in ["Darknet", "CloudDDoS", "IIoTAPT"]:
        args.gpuname = 0
        args.modelname = 'NTDyG_V1'
        args.datatype = '基于正常的异常检测'
        args.extype = 'khop_1'
        args.n_layer = 3
        ds_output_dir = os.path.join(args.output_dir, args.data_set)
        model_dir = os.path.join(ds_output_dir, 'models')
        log_dir = os.path.join(ds_output_dir, 'logs')
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        # 日志文件：记录训练过程
        # log_path = os.path.join(log_dir, f'{config.data_set}{config.modelname}.log')
        log_path = os.path.join(log_dir, f'{args.data_set}{args.modelname}{args.datatype}_{args.extype}.log')
        logger = setup_logger(log_path)
        logger.info(f"开始训练数据集: {args.data_set}")

        if args.data_set == 'Darknet':
            args.input_dim = 64
            args.num_classes = 2
            args.n_heads = 4
            args.hidden_dim = 128
            args.drop_out = 0.2
            args.n_layer = 2
            args.learning_rate = 1e-3
            args.batch_size = 64
            args.weight_decay = 1e-5
            args.edge_dropout = 0.1

        elif args.data_set == 'CloudDDoS':
            args.input_dim = 314
            args.num_classes = 2
            args.n_heads = 2
            args.hidden_dim = 256
            args.drop_out = 0.2
            args.n_layer = 2
            args.learning_rate = 5e-4
            args.batch_size = 32
            args.weight_decay = 1e-5
            args.edge_dropout = 0.1

        elif args.data_set == 'IIoTAPT':
            args.input_dim = 52
            args.num_classes = 2
            args.n_heads = 4
            args.hidden_dim = 128
            args.drop_out = 0.2
            args.n_layer = 2
            args.learning_rate = 1e-3
            args.batch_size = 64
            args.weight_decay = 1e-5
            args.edge_dropout = 0.1


        params_nni = {
            'seed': args.seed,
            'batch_size': args.batch_size,
            'w_time': 0.01,
            'n_heads': args.n_heads,
            'drop_out': args.drop_out,
            'n_layer': args.n_layer,
            'hidden_dim': args.hidden_dim,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'edge_dropout': args.edge_dropout,
            'task': 'Node'
        }

        set_seed(params_nni['seed']) # 就是把 NNI （Neural Network Intelligence）给定的种子参数 params_nni['seed'] 传给我们前面定义的 set_seed 函数，从而在整个训练／评估流程里固定所有的随机性。

        device = torch.device("cuda:{}".format(args.gpuname) if torch.cuda.is_available() else "cpu")

        train_path = '{}/{}_train_{}_{}.pt'.format(args.dir_data, args.data_set, args.datatype, args.extype)
        test_path = '{}/{}_test_{}_{}.pt'.format(args.dir_data, args.data_set, args.datatype, args.extype)
        pad_spec_path = '{}/{}_pad_spec_{}_{}.pt'.format(args.dir_data, args.data_set, args.datatype, args.extype)

        # 只有 train/test/pad_spec 都存在时，才直接加载缓存
        if os.path.exists(train_path) and os.path.exists(test_path) and os.path.exists(pad_spec_path):
            dataset_train = torch.load(train_path, weights_only=False)
            logger.info(f"训练数据集加载完成: {args.data_set}")

            dataset_test = torch.load(test_path, weights_only=False)
            logger.info(f"测试数据集加载完成: {args.data_set}")

            pad_spec = torch.load(pad_spec_path, weights_only=False)
            logger.info(f"统一 padding 规格加载完成: {args.data_set}")

        else:
            dm = dataset.DygDataModule(args)

            # 直接取训练/测试 Dataset 与统一规格
            dataset_train = dm.train_dataset
            dataset_test = dm.test_dataset
            pad_spec = dm.pad_spec

            # 直接缓存“训练用的数据”
            # torch.save(dataset_train, train_path, _use_new_zipfile_serialization=False)
            # torch.save(dataset_test, test_path, _use_new_zipfile_serialization=False)
            # torch.save(pad_spec, pad_spec_path, _use_new_zipfile_serialization=False)

            logger.info(f"训练数据集已生成并缓存: {train_path}")
            logger.info(f"测试数据集已生成并缓存: {test_path}")
            logger.info(f"统一 padding 规格已生成并缓存: {pad_spec_path}")

        # 用缓存/生成得到的统一规格构造 collate_fn
        collate_fn = dataset.UnifiedCollate(pad_spec)

        loader_train = torch.utils.data.DataLoader(
            dataset=dataset_train,
            batch_size=args.batch_size,
            shuffle=True,          # 训练集建议打乱
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_fn.dyg_collate_fn
        )

        loader_test = torch.utils.data.DataLoader(
            dataset=dataset_test,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_fn.dyg_collate_fn
        )


        GNN = CensNet(17, args.input_dim, args.drop_out)
        # transformer = TransformerBinaryClassifier(config, device, hidden_size=config.hidden_dim)
        transformer = TransformerTargetEdgeClassifier(args, device, hidden_size=args.hidden_dim)
  
        backbone = CombinedModel(args, GNN, transformer) # 组合

        model = backbone.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

        # model_path = os.path.join(model_dir, f'final_{args.data_set}_{args.modelname}_{args.datatype}_{args.extype}.pkl')
        # if os.path.exists(model_path):
        #     model.load_state_dict(torch.load(model_path, map_location=device))
        #     model.to(device)

        # =========================
        # 4 个 IF 模型
        # =========================
        # （2）Transformer结果 + IF
        model_if_logit = IsolationForest(
            n_estimators=200, max_samples=1024, contamination=0.03,
            random_state=42, n_jobs=-1
        )
        # （3）Transformer表征 + IF
        model_if_feat = IsolationForest(
            n_estimators=200, max_samples=1024, contamination=0.03,
            random_state=42, n_jobs=-1
        )
        # （4）GRU结果 + Transformer结果 + IF
        model_if_logit_gru = IsolationForest(
            n_estimators=200, max_samples=1024, contamination=0.03,
            random_state=42, n_jobs=-1
        )
        # （5）GRU结果 + Transformer表征 + IF
        model_if_feat_gru = IsolationForest(
            n_estimators=200, max_samples=1024, contamination=0.03,
            random_state=42, n_jobs=-1
        )

        # =========================
        # 各方案最优模型保存路径
        # =========================
        model_path_direct = os.path.join(
            model_dir,
            f'Finetune_direct_{args.data_set}_{args.modelname}_{args.datatype}_{args.extype}.pkl'
        )
        model_path_logit_if = os.path.join(
            model_dir,
            f'Finetune_logitIF_{args.data_set}_{args.modelname}_{args.datatype}_{args.extype}.pkl'
        )
        model_path_feat_if = os.path.join(
            model_dir,
            f'Finetune_featIF_{args.data_set}_{args.modelname}_{args.datatype}_{args.extype}.pkl'
        )
        model_path_logit_gru_if = os.path.join(
            model_dir,
            f'Finetune_logitGRUIF_{args.data_set}_{args.modelname}_{args.datatype}_{args.extype}.pkl'
        )
        model_path_feat_gru_if = os.path.join(
            model_dir,
            f'Finetune_featGRUIF_{args.data_set}_{args.modelname}_{args.datatype}_{args.extype}.pkl'
        )

        # =========================
        # 5 组结果的最优指标
        # 每组保存: AUC / ACC / PRE / REC / F1
        # =========================
        best_direct = {"auc": 0.0, "acc": 0.0, "prec": 0.0, "rec": 0.0, "f1": 0.0}
        best_logit_if = {"auc": 0.0, "acc": 0.0, "prec": 0.0, "rec": 0.0, "f1": 0.0}
        best_feat_if = {"auc": 0.0, "acc": 0.0, "prec": 0.0, "rec": 0.0, "f1": 0.0}
        best_logit_gru_if = {"auc": 0.0, "acc": 0.0, "prec": 0.0, "rec": 0.0, "f1": 0.0}
        best_feat_gru_if = {"auc": 0.0, "acc": 0.0, "prec": 0.0, "rec": 0.0, "f1": 0.0}

        loss_fn = nn.CrossEntropyLoss()
        early_stopper = EarlyStopMonitor(max_round=15, higher_better=True)

        for epoch in range(args.n_epochs):
            logger.info(f"[{args.data_set}] Epoch {epoch}/{args.n_epochs} 开始")

            model.train()

            with tqdm(total=len(loader_train)) as t:
                for batch_idx, batch_sample in enumerate(loader_train):

                    # ===== 读取当前 batch =====
                    input_nodes_feature = batch_sample["input_nodes_feature"]
                    input_edges_feature = batch_sample["input_edges_feature"]
                    Tmats = batch_sample["Tmats"]
                    adjs = batch_sample["adjs"]
                    eadjs = batch_sample["eadjs"]
                    edge_mask = batch_sample["edge_mask"]
                    labels = batch_sample["labels"]
                    num_nodes = batch_sample["num_nodes"]
                    num_edges = batch_sample["num_edges"]
                    target_edge_index = batch_sample["target_edge_index"]

                    def ensure_dense(tensor):
                        if hasattr(tensor, "is_sparse") and tensor.is_sparse:
                            return tensor.to_dense()
                        return tensor

                    input_nodes_feature = ensure_dense(input_nodes_feature.to(device))
                    input_edges_feature = ensure_dense(input_edges_feature.to(device))
                    Tmats = ensure_dense(Tmats.to(device))
                    adjs = ensure_dense(adjs.to(device))
                    eadjs = ensure_dense(eadjs.to(device))
                    edge_mask = ensure_dense(edge_mask.to(device))

                    labels = labels.to(device)
                    num_nodes = num_nodes.to(device)
                    num_edges = num_edges.to(device)
                    target_edge_index = target_edge_index.to(device)

                    t.set_description(f"Epoch {epoch}")
                    optimizer.zero_grad()

                    logits, _ = model(
                        input_nodes_feature,
                        input_edges_feature,
                        input_edges_feature,   # H_e_pad（兼容旧接口）
                        eadjs,
                        adjs,
                        Tmats,
                        edge_mask,
                        num_nodes=num_nodes,
                        num_edges=num_edges,
                        target_edge_index=target_edge_index
                    )

                    y1 = labels.long()
                    idx_local = torch.arange(logits.size(0), device=logits.device)

                    loss = 0.5 * loss_fn(logits, y1) + 0.5 * bpr_loss_binary_normal_only(
                        logits, y1, idx_local,
                        intra_margin=0.05,
                        pseudo_margin=0.10,
                        lambda_pseudo=0.20,
                        hard_ratio=0.20
                        )

                    loss.backward()
                    optimizer.step()

                    t.set_postfix(loss=float(loss.item()))
                    t.update(1)

            # =========================
            # 训练集输出，用于拟合 IF
            # logits_train: Transformer结果
            # y_train:      标签
            # mean_out:     Transformer表征
            # gru_train:    GRU异常分数
            # =========================
            logits_train, y_train, mean_out, gru_train = eval_epoch_train_finetune(loader_train, model, device)

            # =========================
            # 构造 4 类 IF 的训练输入
            # =========================
            train_logit = logits_train
            train_feat = mean_out
            train_logit_gru = np.concatenate([logits_train, gru_train.reshape(-1, 1)], axis=1)
            train_feat_gru = np.concatenate([mean_out, gru_train.reshape(-1, 1)], axis=1)

            # =========================
            # 拟合 4 个 IF
            # =========================
            model_if_logit.fit(train_logit)
            model_if_feat.fit(train_feat)
            model_if_logit_gru.fit(train_logit_gru)
            model_if_feat_gru.fit(train_feat_gru)

            # =========================
            # 在测试集上评估 5 组结果
            # 这里要求你对应修改 eval_epoch_finetune_one：
            # 返回 5 组结果，每组 5 个指标
            # =========================
            (
                direct_auc, direct_acc, direct_prec, direct_rec, direct_f1,
                logit_if_auc, logit_if_acc, logit_if_prec, logit_if_rec, logit_if_f1,
                feat_if_auc, feat_if_acc, feat_if_prec, feat_if_rec, feat_if_f1,
                logit_gru_if_auc, logit_gru_if_acc, logit_gru_if_prec, logit_gru_if_rec, logit_gru_if_f1,
                feat_gru_if_auc, feat_gru_if_acc, feat_gru_if_prec, feat_gru_if_rec, feat_gru_if_f1
            ) = eval_epoch_finetune_one(
                loader_test,
                model,
                model_if_logit,
                model_if_feat,
                model_if_logit_gru,
                model_if_feat_gru,
                device
            )

            logger.info(f"[{args.data_set}]")
            logger.info(f"loss: {loss}")

            logger.info(
                f"（1）Transformer直接输出 当前Epoch {epoch} "
                f"AUC: {direct_auc} ACC: {direct_acc} PRE: {direct_prec} REC: {direct_rec} F1: {direct_f1}"
            )
            logger.info(
                f"（2）Transformer结果+IF 当前Epoch {epoch} "
                f"AUC: {logit_if_auc} ACC: {logit_if_acc} PRE: {logit_if_prec} REC: {logit_if_rec} F1: {logit_if_f1}"
            )
            logger.info(
                f"（3）Transformer表征+IF 当前Epoch {epoch} "
                f"AUC: {feat_if_auc} ACC: {feat_if_acc} PRE: {feat_if_prec} REC: {feat_if_rec} F1: {feat_if_f1}"
            )
            logger.info(
                f"（4）GRU结果+Transformer结果+IF 当前Epoch {epoch} "
                f"AUC: {logit_gru_if_auc} ACC: {logit_gru_if_acc} PRE: {logit_gru_if_prec} REC: {logit_gru_if_rec} F1: {logit_gru_if_f1}"
            )
            logger.info(
                f"（5）GRU结果+Transformer表征+IF 当前Epoch {epoch} "
                f"AUC: {feat_gru_if_auc} ACC: {feat_gru_if_acc} PRE: {feat_gru_if_prec} REC: {feat_gru_if_rec} F1: {feat_gru_if_f1}"
            )

            # =========================
            # 更新最优结果 & 保存 backbone
            # 这里只保存 backbone；
            # 如果你后面还要复现 IF 结果，建议同时 pickle 对应 IF 模型
            # =========================
            if direct_auc > best_direct["auc"]:
                best_direct.update({
                    "auc": direct_auc, "acc": direct_acc, "prec": direct_prec,
                    "rec": direct_rec, "f1": direct_f1
                })
                torch.save(model.state_dict(), model_path_direct)

            if logit_if_auc > best_logit_if["auc"]:
                best_logit_if.update({
                    "auc": logit_if_auc, "acc": logit_if_acc, "prec": logit_if_prec,
                    "rec": logit_if_rec, "f1": logit_if_f1
                })
                torch.save(model.state_dict(), model_path_logit_if)

            if feat_if_auc > best_feat_if["auc"]:
                best_feat_if.update({
                    "auc": feat_if_auc, "acc": feat_if_acc, "prec": feat_if_prec,
                    "rec": feat_if_rec, "f1": feat_if_f1
                })
                torch.save(model.state_dict(), model_path_feat_if)

            if logit_gru_if_auc > best_logit_gru_if["auc"]:
                best_logit_gru_if.update({
                    "auc": logit_gru_if_auc, "acc": logit_gru_if_acc, "prec": logit_gru_if_prec,
                    "rec": logit_gru_if_rec, "f1": logit_gru_if_f1
                })
                torch.save(model.state_dict(), model_path_logit_gru_if)

            if feat_gru_if_auc > best_feat_gru_if["auc"]:
                best_feat_gru_if.update({
                    "auc": feat_gru_if_auc, "acc": feat_gru_if_acc, "prec": feat_gru_if_prec,
                    "rec": feat_gru_if_rec, "f1": feat_gru_if_f1
                })
                torch.save(model.state_dict(), model_path_feat_gru_if)

            # 早停：这里仍然以（1）Transformer直接输出的 AUC 为准
            if early_stopper.early_stop_check(best_direct["auc"]):
                print('No improvment over {} epochs, stop training'.format(early_stopper.max_round))
                break

        logger.info(f"[{args.data_set}]")
        logger.info(
            f"（1）Transformer直接输出 最终 "
            f"AUC: {best_direct['auc']} ACC: {best_direct['acc']} PRE: {best_direct['prec']} "
            f"REC: {best_direct['rec']} F1: {best_direct['f1']}"
        )
        logger.info(
            f"（2）Transformer结果+IF 最终 "
            f"AUC: {best_logit_if['auc']} ACC: {best_logit_if['acc']} PRE: {best_logit_if['prec']} "
            f"REC: {best_logit_if['rec']} F1: {best_logit_if['f1']}"
        )
        logger.info(
            f"（3）Transformer表征+IF 最终 "
            f"AUC: {best_feat_if['auc']} ACC: {best_feat_if['acc']} PRE: {best_feat_if['prec']} "
            f"REC: {best_feat_if['rec']} F1: {best_feat_if['f1']}"
        )
        logger.info(
            f"（4）GRU结果+Transformer结果+IF 最终 "
            f"AUC: {best_logit_gru_if['auc']} ACC: {best_logit_gru_if['acc']} PRE: {best_logit_gru_if['prec']} "
            f"REC: {best_logit_gru_if['rec']} F1: {best_logit_gru_if['f1']}"
        )
        logger.info(
            f"（5）GRU结果+Transformer表征+IF 最终 "
            f"AUC: {best_feat_gru_if['auc']} ACC: {best_feat_gru_if['acc']} PRE: {best_feat_gru_if['prec']} "
            f"REC: {best_feat_gru_if['rec']} F1: {best_feat_gru_if['f1']}"
        )
if __name__ == "__main__":
    main()
