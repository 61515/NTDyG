import argparse

parser = argparse.ArgumentParser('IGTraffic Training')
parser.add_argument('--dir_data', type=str, default='./dataset')
parser.add_argument('--output_dir', type=str, default='./outputs',
                        help="保存模型和日志的根目录")
parser.add_argument('--data_set', type=str, default='Darknet')

##data param
parser.add_argument('--batch_size', type=int, default=1024)  # 256
parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--gpus', type=int, default=1)

##model param
parser.add_argument('--input_dim', type=int, default=128)
parser.add_argument('--hidden_dim', type=int, default=258)  # 354
parser.add_argument('--n_heads', type=int, default=4)
parser.add_argument('--drop_out', type=float, default=0.3)
parser.add_argument('--n_layer', type=int, default=5, help='Number of network layers')
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--seed', type=int, default=95540)

parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
parser.add_argument('--emb_size', type=int, default=64)
parser.add_argument('--edge_dropout', type=float, default=0.5)
parser.add_argument('--weight_decay', type=float, default=1e-4)

args = parser.parse_args()
