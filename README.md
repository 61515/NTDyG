# NTDyG

Code for **Memory-Enhanced Temporal Dynamic Graph Learning for IoT Traffic Anomaly Detection**.

This release provides the NTDyG implementation, training and ablation
entrypoints, preprocessing, evaluation, model configurations, split indices,
and processed IIoTAPT/Darknet data. The benchmark and validation-selected
entrypoints use different protocols; see
[reproducibility details](docs/REPRODUCIBILITY.md).

中文说明见 [README_zh.md](README_zh.md).

## Installation

Use Python 3.11. Install the PyTorch build appropriate for your CPU/CUDA system,
then install the remaining dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`requirements.txt` gives supported dependency ranges. GPU process-memory
profiling additionally uses `requirements-runtime.txt`.

## Quick check without traffic data

Run these commands from the repository root:

```bash
python scripts/run_tests.py
python scripts/smoke_test.py --variants full A1 A2 A3 --output-dir outputs/smoke
```

The smoke test creates 120 **synthetic** events, generates a disjoint 60/20/20
split, trains memory and classifier models for one epoch each, and independently
recomputes the test metrics for all four variants. It checks execution; its scores
are not paper results. Select a new output directory for a second invocation.

## Data

Processed IIoTAPT and Darknet event tables are included as four gzip archives
in `data/archives/`, covering the binary and limited-label schemas (42.34 MB
compressed; 170.81 MB expanded). Unpack and verify them before training:

```bash
python scripts/unpack_data.py
```

For binary data only, add `--schema binary`. The helper uses only the Python
standard library and verifies the original CSV hashes. Each gzip file is below
GitHub's 25 MiB browser-upload limit. Dataset citations and provider reuse
statements are in [docs/DATA_ATTRIBUTION.md](docs/DATA_ATTRIBUTION.md).

For example, train with the included IIoTAPT binary table:

```bash
python code/run_candidate_experiment.py --dataset IIoTAPT --ablation full \
  --data-dir data/binary --split-dir splits/ordered --output-dir outputs/iiot_full
```

Original source traffic is available from its providers:

- [BCCC-cPacket-Cloud-DDoS-2024](https://www.yorku.ca/research/bccc/ucs-technical/cybersecurity-datasets-cds/cloud-ddos-attacks-bccc-cpacket-cloud-ddos-2024/)
- [IIoT Dataset 2024](https://www.unb.ca/cic/datasets/iiot-dataset-2024.html)
- [CIC-Darknet2020](https://www.unb.ca/cic/datasets/darknet2020.html)

CloudDDoS traffic is not bundled because redistribution permission was not
established from the inspected BCCC provider pages. Trained weights and
generated feature caches are also not bundled. The processed binary event
format is documented in [docs/DATA.md](docs/DATA.md). Dataset names alone do
not establish the row ordering required by the saved split indices.

For CloudDDoS or another CSV already matching the retained processed files, place
`CloudDDoS_re.csv`, `IIoTAPT_re.csv`, and/or `Darknet_re.csv` in `data/binary/`.
The supplied split indices refer to the exact checksums in
`configs/data_sources.json`.

For a new flow CSV, explicitly identify endpoints, labels, numeric ordering,
and flow columns. For example, using the retained IIoTAPT feature names:

```bash
python scripts/prepare_data.py \
  --input /path/to/flows.csv --dataset IIoTAPT \
  --source-column "Src IP Label" --destination-column "Dst IP Label" \
  --label-column Label_encoded --normal-label 0 --positive-label 1 \
  --order-column Timestamp_Label \
  --output-dir data/custom_binary --split-dir splits/custom
```

Use `--use-row-order` only when source-row order is the intended event order.
This conversion creates a new dataset and new splits; it does not reconstruct
the historical CloudDDoS sampling or recover unavailable raw time annotations.

## Validation-selected NTDyG

```bash
python code/run_candidate_experiment.py \
  --dataset CloudDDoS --ablation full --seed 95545 --device cpu \
  --data-dir data/binary --split-dir splits/ordered \
  --memory-epochs 5 --memory-patience 3 \
  --model-epochs 20 --patience 5 --output-dir outputs/cloud_full
```

Select `--device cuda:0` for a CUDA classifier. Endpoint-memory training runs on
CPU, matching the saved ablation implementation. For newly converted data,
replace the data and split paths with those printed by `prepare_data.py`.

`--ablation` accepts:

| Variant | Change |
| --- | --- |
| `full` | Complete model |
| `A1` | Symmetric endpoint-memory message inputs |
| `A2` | Mask node-memory coordinates before graph encoding |
| `A3` | Mask graph context before graph–flow fusion |

Every run freshly trains memory and classifier weights. Memory training uses
normal training events and training entities for negative sampling. The
classifier uses labeled training events. Both stages select by validation AP;
validation F1 selects an alert threshold, and the selected classifier evaluates
the test set once. Scoring uses the positive-class logit without Isolation Forest.

Run outputs contain model checkpoints, epoch metrics, memory-selection history,
validation/test scores, source checksum, feature-column order, and configuration.
Existing run directories are never overwritten.

```bash
python code/summarize_runs.py --runs-root outputs/cloud_full --output outputs/cloud_summary.csv
python code/evaluate_scores.py \
  --scores outputs/cloud_full/candidate_60_20_20_CloudDDoS_full_seed_95545/test_scores.csv \
  --run-results outputs/cloud_full/candidate_60_20_20_CloudDDoS_full_seed_95545/results.json \
  --output outputs/recomputed_metrics.json
```

The evaluation command reads the frozen validation threshold from that run's
`results.json`; it does not select a threshold on the supplied test scores.

## Other experiments

- [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md): temporal baselines, limited-label
  classification, fixed-memory legacy entrypoints, feature masks, and streaming
  resource profiling.
- [configs/model_config.json](configs/model_config.json): dataset-specific
  widths, heads, batch sizes, learning rates, and classifier settings.
- [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md): vendored DyGLib source and
  its MIT license.

## Release scope

The package supports fresh training with the supplied schema. Existing
experiment results, per-run scores, logs, checkpoints, and internal development
records are excluded. The fixed-memory benchmark wrapper requires external
memory inputs. See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the
entrypoints' scope and limitations.
