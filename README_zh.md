# NTDyG 代码发布包

对应论文：**Memory-Enhanced Temporal Dynamic Graph Learning for IoT Traffic Anomaly Detection**。

本目录可作为 GitHub 仓库根目录上传，包含：

- NTDyG 模型、端点记忆、历史图提取、图流融合和分类器代码。
- 基于已保存 CloudDDoS 消融代码快照的训练入口，支持 Full/A1/A2/A3。
- GraphMixer、DyGFormer、TGAT、CAWN 的旧协议对比入口及 DyGLib 来源和许可证。
- 通用流记录预处理、划分生成、指标重算、多次运行汇总和流式计时脚本。
- 三个数据集的特征列、模型配置和已保存划分索引。
- IIoTAPT、Darknet 的二分类与多分类处理后数据压缩包，以及校验解压脚本。
- 安装依赖、运行示例、现有测试及合成数据流程检查。

## 安装与快速检查

建议使用 Python 3.11。先安装适合本机的 PyTorch，再在此目录执行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/run_tests.py
python scripts/smoke_test.py --variants full A1 A2 A3 --output-dir outputs/smoke
```

最后一条命令只使用120条合成事件，检查预处理、划分、记忆训练、构图、分类器训练和指标重算。它不是论文实验结果。

## 正式数据与训练

已包含 IIoTAPT、Darknet 的二分类和多分类处理后 CSV，共四个 gzip 压缩文件，合计 42.34 MB，解压后 170.81 MB。每个压缩文件均小于 GitHub 网页上传的 25 MiB 限制。先在本目录执行：

```bash
python scripts/unpack_data.py
```

脚本将文件放入 `data/binary/` 和 `data/multiclass/`，并校验其与原实验 CSV 的 SHA-256 一致。仅需二分类数据时，加上 `--schema binary`。上传时保留 `data/archives/` 中的压缩文件，解压后的 CSV 由 `.gitignore` 排除。

IIoTAPT、Darknet 的提供方允许注明来源后再分发；来源和引用见 [docs/DATA_ATTRIBUTION.md](docs/DATA_ATTRIBUTION.md)。CloudDDoS 官网未找到明确再分发条款，本公开包保留官方获取链接、字段和校验值，暂不附带该数据。原始流量包、训练权重、缓存及完整日志也未附带。CSV 字段和索引适用范围见 [docs/DATA.md](docs/DATA.md)。

若另行取得与本项目一致的 `CloudDDoS_re.csv`，将其放入 `data/binary/`。使用已保存索引前，文件需与 `configs/data_sources.json` 中的哈希一致。

```bash
python code/run_candidate_experiment.py \
  --dataset CloudDDoS --ablation full --seed 95545 --device cpu \
  --data-dir data/binary --split-dir splits/ordered \
  --memory-epochs 5 --memory-patience 3 \
  --model-epochs 20 --patience 5 --output-dir outputs/cloud_full
```

`--ablation` 可设为 `full`、`A1`、`A2`、`A3`；分类器可使用 `--device cuda:0`。
各次运行重新训练记忆与分类器，两个阶段均按验证集 AP 选模，以验证集 F1 选择阈值，最后评估一次测试集。该入口与主表旧协议的固定记忆分类器重复不同。

新CSV可使用 `scripts/prepare_data.py` 明确指定端点、类别、特征和顺序列，再使用其生成的新索引。该脚本没有补造原始数据的抽样过程，也不会把行序号当作真实日历时间。

更多运行命令见英文 [README.md](README.md) 和 [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)。

## 发布范围

本包提供模型代码、预处理与评估脚本、配置、划分索引和可再分发的数据。已有实验结果、逐次指标、得分文件、运行日志、模型权重和内部开发核验材料均不包含在本包中。测试目录使用合成输入检查代码行为，不附带本项目的私有实验输出。

固定记忆旧协议入口需要另行提供记忆输入；各入口适用范围见 [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)。

## 上传

上传本目录中的内容，并包含 `.gitignore` 和 `data/archives/`。网页上传时先解压外层发布 ZIP，再上传目录中的文件。实验输出、日志、权重、内部材料和认证凭据不属于发布内容；`.gitignore` 已加入相应排除规则。
