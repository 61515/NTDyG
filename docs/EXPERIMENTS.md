# Experiment entrypoints

Run commands from the repository root. `NTDYG_DATA_DIR` and
`NTDYG_SPLIT_DIR` override the default binary data and ordered-split directories.
The main runner also accepts `--data-dir` and `--split-dir` directly.

## Parameter-matched controls

`code/run_candidate_experiment.py` defaults to five memory epochs (patience 3),
twenty classifier epochs (patience 5), and ordered classifier batches. Both
stages use validation AP for checkpoint selection. The Full/A1/A2/A3 controls
retain the same registered parameter count. Each run writes its random seed,
source checksum, configuration, per-epoch metrics, and validation/test scores.

`summarize_runs.py` includes every completed run supplied to it. Run outputs
are generated locally; existing project experiment outputs are not included.

## Temporal graph baselines on the retained binary protocol

With exact retained binary CSVs in `data/binary/`:

```bash
python code/run_legacy_temporal_baseline.py \
  --dataset IIoTAPT --model TGAT --seed 95545 --device cpu \
  --epochs 200 --output-dir outputs/retained_temporal
```

`--model` accepts `GraphMixer`, `DyGFormer`, `TGAT`, or `CAWN`. CAWN supports
`--cawn-indexed-position`; existing tests compare the indexed implementation's
outputs and gradients with its upstream implementation. Each encoder uses a
directed event-classification head over endpoint embeddings and flow features.
The runner retains normal-only training, shuffled targets, existing train/test
indices, and **test-AUC** checkpoint selection. It has no validation split and
must not be described as an independent validation-selected comparison.

The source uses vendored DyGLib encoders with their license and upstream commit.
`NTDYG_RETAINED_SPLIT_DIR` overrides `splits/retained/`.

## Limited-label temporal baselines

Obtain the separate multiclass-schema CSVs described in `data/multiclass/README.md`:

```bash
python code/run_legacy_multiclass.py --prepare --output-dir outputs/limited_label
python code/run_legacy_multiclass.py \
  --output-dir outputs/limited_label --dataset CloudDDoS --model GraphMixer \
  --rate 1 --seed 95545 --device cpu --epochs 200
```

Preparation saves nested 1%/5% uniform training draws, existing test indices,
source fingerprints, and the frozen protocol. There are 26/2/8 classes for
CloudDDoS/IIoTAPT/Darknet. `NTDYG_MULTICLASS_DATA_DIR` overrides data location;
`NTDYG_MULTICLASS_SPLIT_DIR` overrides the retained test-index directory.

## Retained NTDyG classifier/Isolation Forest repetitions

`code/run_ntdyg_original_entrypoint.py` and `legacy/binary/` retain the original
classifier and its IF branches. They require the fixed memory input files
listed in `data/legacy_memory/README.md`, which are not bundled. With those
inputs available, run:

```bash
python code/run_ntdyg_original_entrypoint.py prepare --root outputs/retained_ntdyg
python code/run_ntdyg_original_entrypoint.py pipeline \
  --root outputs/retained_ntdyg --dataset IIoTAPT --seed 95545 --gpu 0
```

`NTDYG_MEMORY_INPUT_DIR` selects an external fixed-memory input directory.
Fresh memory training produces different inputs from a fixed-memory benchmark.
Invoke the legacy source files through the wrapper, which parameterizes their
dataset loop.

## Missing-channel diagnostic

After a completed validation-selected Full run:

```bash
python code/evaluate_feature_corruption.py \
  --dataset IIoTAPT \
  --run-dir outputs/iiot_full/candidate_60_20_20_IIoTAPT_full_seed_95545 \
  --output-dir outputs/iiot_corruption --device cpu --include-clean
```

The helper uses nested masks on standardized channels and the fixed clean
validation threshold. It does not retune on the corrupted test examples.

## Event-to-score execution

```bash
python code/benchmark_stream.py \
  --dataset CloudDDoS \
  --run-dir outputs/cloud_full/candidate_60_20_20_CloudDDoS_full_seed_95545 \
  --output-dir outputs/stream_profile --device cpu --verify-events 128
python code/benchmark_stream.py \
  --dataset CloudDDoS \
  --run-dir outputs/cloud_full/candidate_60_20_20_CloudDDoS_full_seed_95545 \
  --output-dir outputs/stream_profile --device cpu --repeat 1
```

The preflight compares incremental scores with the saved batch evaluation. The
full measurement includes standardization, observed-ID registration, memory
computation, causal graph construction, batch-one classification, thresholding,
and state/history commit. Capture, flow extraction, CSV loading, and model loading
are excluded. Memory remains on CPU. For CUDA classifier and GPU process-memory
profiling, install `requirements-runtime.txt` and use `--device cuda:0`. Use
distinct repeat numbers and report variation across repetitions. Hardware differs
across systems, so the measured values are not expected to match the paper exactly.

## Code checks

`scripts/run_tests.py` runs the retained semantic tests. The synthetic smoke test
also independently recomputes ROC-AUC, AP, and F1 using scikit-learn. These
commands create their own local outputs; no existing test-result records are
included in this release.
