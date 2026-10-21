# Entrypoint scope

This release contains source code, model configurations, preprocessing and
evaluation utilities, row-index splits, and processed IIoTAPT/Darknet data.
Existing experimental results, run scores, logs, checkpoints, and internal
development records are excluded.

## Validation-selected training

`code/run_candidate_experiment.py` trains endpoint memory and the classifier
afresh. Both stages select checkpoints by validation AP. Validation F1 selects
the threshold, and the selected classifier evaluates the test set once.
Scaling is fitted on training rows. Memory training uses normal training
events and training-only entities for negative sampling. New endpoint slots
start at zero, and observed traffic can update state after scoring.

The Full/A1/A2/A3 variants use the same registered parameter count. The public
defaults use five memory epochs and twenty classifier epochs, with early
stopping. Numerical results depend on the supplied CSV, configuration, random
initialization, and hardware.

## Retained benchmark adapters

The legacy adapters preserve their source protocols, including test-AUC
checkpoint selection and some overlapping train/test indices. Their scores
must not be interpreted as independent, untouched test estimates. The
limited-label adapter creates new nested training draws; it does not recover
unavailable historical sampling decisions.

The fixed-memory NTDyG wrapper requires external pretrained memory weights
and states. These inputs are not included. Fresh memory training produces
different inputs, so the release does not guarantee exact reconstruction of
every historical paper result.

## Data and event order

The included IIoTAPT and Darknet gzip archives preserve the original processed
CSV bytes. `configs/processed_data_archives.json` records archive and CSV
checksums. `scripts/unpack_data.py` verifies these before training. The saved
row indices apply to the exact corresponding CSVs.

CloudDDoS tables are not included. The generic CSV converter requires explicit
field and label mappings; it does not reconstruct original packet extraction
or historical sampling. Event-order keys in the processed tables are not
verified calendar timestamps.

## Local checks

`scripts/run_tests.py` uses small fixtures to check code behavior.
`scripts/smoke_test.py` uses synthetic events for a short end-to-end run.
These checks do not substitute for full training or establish replication of
paper results. Their outputs are generated locally and excluded by `.gitignore`.
