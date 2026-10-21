import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.nn.functional as F
from option import args
from utils import setup_logger, EarlyStopMonitor, eval_epoch_train, set_seed, alignment_infonce
from utils import eval_epoch_one, bpr_loss_multiclass, eval_epoch_train_finetune, eval_epoch_finetune_one, bpr_loss_multiclass_safe
import data_pre as dataset

from model.CensNet import CensNet
from model.Combine import CombinedModel

from model.Transformer import TransformerBinaryClassifier, TransformerMultiClassClassifier
import numpy as np
from sklearn.ensemble import IsolationForest

from tqdm import tqdm
def main():
    for args.data_set in ["Darknet"]:
    # for args.data_set in ["Darknet", "CloudDDoS", "IIoTAPT"]:
        args.gpuname = 2
        args.modelname = 'NTDyG_V1'
        args.datatype = '基于正常的异常检测'
        args.extype = 'Test_1'
        args.n_layer = 5
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
            args.input_dim = 64 # (79-15)
            args.num_classes = 2
            args.n_heads = 8
        elif args.data_set == 'CloudDDoS':
            args.input_dim = 314
            args.num_classes = 2 #8+18
            args.n_heads = 2
        elif args.data_set == 'IIoTAPT':
            args.input_dim = 52
            args.num_classes = 2
            args.n_heads = 4


        params_nni = {
            'seed': args.seed,
            'batch_size': args.batch_size,
            'w_time': 0.01,
            'n_heads': args.n_heads,
            'drop_out': args.drop_out,
            'n_layer': args.n_layer,
            'task': 'Node'
        }

        set_seed(params_nni['seed']) # 就是把 NNI （Neural Network Intelligence）给定的种子参数 params_nni['seed'] 传给我们前面定义的 set_seed 函数，从而在整个训练／评估流程里固定所有的随机性。

        device = torch.device("cuda:{}".format(args.gpuname) if torch.cuda.is_available() else "cpu")
        train_path = '{}/{}_train_{}.pt'.format(args.dir_data, args.data_set, args.datatype)
        test_path = '{}/{}_test_{}.pt'.format(args.dir_data, args.data_set, args.datatype)

        if os.path.exists(test_path):
            dataset_test = torch.load(test_path, weights_only=False)
            logger.info(f"测试数据集加载完成: {args.data_set}")
        else:
            dataset_test = dataset.DygDataset(args, 'test', train_path, test_path)
            torch.save(dataset_test, test_path, _use_new_zipfile_serialization=False)
            print("生成并缓存测试数据")
            
        if os.path.exists(train_path):
            dataset_train = torch.load(train_path, weights_only=False)
            logger.info(f"训练数据集加载完成: {args.data_set}")
        else:
            dataset_train = dataset.DygDataset(args, 'train', train_path, test_path)
            torch.save(dataset_train, train_path, _use_new_zipfile_serialization=False)
            print("生成并缓存训练数据")
      

        collate_fn = dataset.Collate(args) # 传入 config 让它可以根据你的配置（比如 padding 大小、特征维度等）做相应的处理。

        GNN = CensNet(1, args.input_dim, args.drop_out)
        # transformer = TransformerBinaryClassifier(config, device, hidden_size=config.hidden_dim)
        transformer = TransformerMultiClassClassifier(args, device, hidden_size=args.hidden_dim)
  
        backbone = CombinedModel(args, GNN, transformer) # 组合

        model = backbone.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

        model_path = os.path.join(model_dir, f'final_{args.data_set}_{args.modelname}_{args.datatype}_{args.extype}.pkl')
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)

        modelx = IsolationForest(n_estimators=200, max_samples=1024, contamination=0.1, random_state=42, n_jobs=-1)
        modelx1 = IsolationForest(n_estimators=200, max_samples=1024, contamination=0.1, random_state=42, n_jobs=-1)
        
        model_path2 = os.path.join(model_dir, f'Finetune_final_{args.data_set}_{args.modelname}_{args.datatype}_{args.extype}.pkl')
        model_path3 = os.path.join(model_dir, f'Finetune_final2_{args.data_set}_{args.modelname}_{args.datatype}_{args.extype}.pkl')
        model_path4 = os.path.join(model_dir, f'Finetune_final2_{args.data_set}_{args.modelname}_{args.datatype}_{args.extype}.pkl')


        loader_train = torch.utils.data.DataLoader(
            dataset=dataset_train, # 1. 要加载的数据集对象，需实现 __len__ 和 __getitem__
            batch_size=args.batch_size, # 2. 每个 batch 中样本的数量
            shuffle=False, # 3. 是否在每个 epoch 开始前打乱数据
            num_workers= 0, # 4. 用多少个子进程来并行加载数据
            pin_memory=False, # 5. 是否把数据加载到页锁定内存（加速 GPU 传输）
            collate_fn=collate_fn.dyg_collate_fn # 6. 自定义的拼 batch 函数，将 list of samples 转成 batch 字典
        )

        loader_test = torch.utils.data.DataLoader(
            dataset=dataset_test,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers= 0,
            collate_fn=collate_fn.dyg_collate_fn
        )   

        max_test_auc = 0.0
        max_test_acc = 0.0
        max_test_prec = 0.0
        max_test_rec = 0.0
        max_test_f1 = 0.0
        max_a1 = 0.0
        max_a2 = 0.0
        max_a3 = 0.0
        max_a4 = 0.0
        max_a5 = 0.0
        max_b1 = 0.0
        max_b2 = 0.0
        max_b3 = 0.0
        max_b4 = 0.0
        max_b5 = 0.0
        loss_fn = nn.CrossEntropyLoss()
  
        early_stopper = EarlyStopMonitor(max_round=15, higher_better=True) # 早停
        # count = 0
        for epoch in range(args.n_epochs):
            logger.info(f"[{args.data_set}] Epoch {epoch}/{args.n_epochs} 开始")
            # auc = []

            with tqdm(total=len(loader_train)) as t:
                # loader_train 是 PyTorch 的 DataLoader，负责按 batch 产出训练样本
                # len(loader_train) 表示训练集中 batch 的数量。
                for batch_idx, batch_sample in enumerate(loader_train):
                    batch_indices = batch_sample['Timestamp_Label']         
                    input_nodes_feature = batch_sample['input_nodes_feature']
                    input_edges_feature = batch_sample['input_edges_feature']
                    input_edges_pad = batch_sample['input_edges_pad']
                    labels2 = batch_sample['labels2']
                    Tmats = batch_sample['Tmats']
                    adjs = batch_sample['adjs']
                    eadjs = batch_sample['eadjs']
                    mask_edge = batch_sample['mask_edge']
                    # 定义一个小函数：若是稀疏张量则 to_dense，否则原样返回
                    def ensure_dense(tensor):
                        if hasattr(tensor, "is_sparse") and tensor.is_sparse:
                            # 打印一下确认来源，便于调试
                            # print(f"Converting sparse tensor to dense: shape={tensor.shape}, layout={tensor.layout}")
                            return tensor.to_dense()
                        return tensor
                    input_nodes_feature = [ensure_dense(tensor.to(device)) for tensor in input_nodes_feature]
                    input_edges_feature = [ensure_dense(tensor.to(device)) for tensor in input_edges_feature]
                    input_edges_pad = ensure_dense(input_edges_pad.to(device))
                    Tmats = [ensure_dense(tensor.to(device)) for tensor in Tmats]
                    adjs = [ensure_dense(tensor.to(device)) for tensor in adjs]
                    eadjs = [ensure_dense(tensor.to(device)) for tensor in eadjs]
                    mask_edge = ensure_dense(mask_edge.to(device))

                    t.set_description('Epoch %i' % epoch) # 将进度条前缀设置为当前 epoch 标号，便于观察。
                    # 也可同时显示更多信息，如学习率等：t.set_description(f'Epoch {epoch}, lr={cur_lr:.5f}')。
                    optimizer.zero_grad() # 清除上一步累积的梯度，确保当前 batch 反向时不混入之前的梯度。
                    model.train()
                    logits, _= model(
                        input_nodes_feature,
                        input_edges_feature,
                        input_edges_pad.to(device),
                        eadjs,
                        adjs,
                        Tmats,
                        mask_edge.to(device)
                    ) # 调用模型的 forward，传入准备好的各输入
                    y = labels2.to(device)
                    y1 = y.long()
                    # loss = 0.5 * loss_fn(logits, y1) + 0.5 * bpr_loss_multiclass(logits, y1, batch_indices.numpy())
                    # idx_local = torch.arange(logits.size(0), device=logits.device)  # 0..B-1
                    idx_local = torch.arange(logits.size(0), device=logits.device)
                    loss = 0.5 * loss_fn(logits, y1) + 0.5 * bpr_loss_multiclass_safe(logits, y1, idx_local)

                    # y = labels2.to(device)
                    # y = y.to(torch.float32)
                    # logits_cpu = logits.detach().cpu().numpy()
                    # logits_2_cpu = logits_2.detach().cpu().numpy()
                    # feat_cpu = feat.detach().cpu().numpy()
                    # y_cpu = y.detach().cpu().numpy()
                    # all_logits.append(logits_cpu)
                    # all_logits_2.append(logits_2_cpu)
                    # all_labels.append(y_cpu)
                    # all_feats.append(feat_cpu)

                    loss.backward()
                    optimizer.step()
                    t.set_postfix(loss=loss)
                    # t.set_postfix(loss=loss)：在进度条末尾显示当前 loss 值，便于实时监控训练情况。
                    t.update(1)

            logits_train, y_train, mean_out = eval_epoch_train_finetune(loader_train, model, device)      
            
            # （1）trans结果

            # （2）只包含特征,trans的特征
            modelx.fit(mean_out, y_train)

            # （3）只包含结果
            modelx1.fit(logits_train, y_train)
    
            test_auc, test_acc, test_prec, test_rec, test_f1, a1, a2, a3, a4, a5, b1, b2, b3, b4, b5= eval_epoch_finetune_one(loader_test, model, modelx, modelx1, device)
            
          
            logger.info(f"[{args.data_set}]")
            logger.info(f"loss: {loss}")
            logger.info(f"trans结果 当前Epoch {epoch} AUC: {test_auc} ACC: {test_acc} PRE: {test_prec} REC: {test_rec} F1: {test_f1}")
            logger.info(f"trans特征+IS 当前Epoch {epoch} AUC: {a1} ACC: {a2} PRE: {a3} REC: {a4} F1: {a5}")
            logger.info(f"trans的结果+IS 当前Epoch {epoch} AUC: {b1} ACC: {b2} PRE: {b3} REC: {b4} F1: {b5}")
            
            if test_auc > max_test_auc:
                max_test_auc = test_auc
                max_test_acc = test_acc
                max_test_prec = test_prec
                max_test_rec = test_rec
                max_test_f1 = test_f1
                
                torch.save(model.state_dict(), model_path2)

            if a1 > max_a1:
                max_a1 = a1
                max_a2 = a2
                max_a3 = a3
                max_a4 = a4
                max_a5 = a5

                torch.save(model.state_dict(), model_path3)

            if b1 > max_b1:
                max_b1 = b1
                max_b2 = b2
                max_b3 = b3
                max_b4 = b4
                max_b5 = b5

                torch.save(model.state_dict(), model_path4)

            if early_stopper.early_stop_check(max_test_auc):  # 注意这里得变一下
                print('No improvment over {} epochs, stop training'.format(early_stopper.max_round))
                break
            else:
                pass

        logger.info(f"[{args.data_set}]")
        logger.info(f"trans结果 最终 AUC: {max_a1} ACC: {max_a2} PRE: {max_a3} REC: {max_a4} F1: {max_a5}")
        logger.info(f"trans特征+IS 最终 AUC: {max_b1} ACC: {max_b2} PRE: {max_b3} REC: {max_b4} F1: {max_b5}")
        logger.info(f"trans的结果+IS 最终 AUC: {max_test_auc} ACC: {max_test_acc} PRE: {max_test_prec} REC: {max_test_rec} F1: {max_test_f1}")




if __name__ == "__main__":
    main()
