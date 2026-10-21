# Data and preprocessing

Processed IIoTAPT and Darknet tables are included in `data/archives/` for both
binary and limited-label experiments. Run `python scripts/unpack_data.py`
from the release root to restore the exact CSVs and verify their hashes.
CloudDDoS tables are not bundled. See [dataset attribution](DATA_ATTRIBUTION.md)
for provider reuse statements and required citations.

The binary-schema files have four metadata columns and numeric flow features:

| Column | Meaning |
| --- | --- |
| `Src IP Label` | Integer source endpoint identifier |
| `Dst IP Label` | Integer destination endpoint identifier |
| `Timestamp_Label` | Numeric event-order key |
| `Label_encoded` | 0: reference/normal; 1: positive class |

The positive class is anomalous traffic on CloudDDoS/IIoTAPT and Tor/VPN traffic
on Darknet. Darknet labels do not establish maliciousness. Feature names and
their order are supplied in `configs/feature_columns.json`: CloudDDoS 314,
IIoTAPT 52, and Darknet 64 features. Labels, endpoint identifiers, and ordering
metadata do not enter the flow-feature matrix.

`Timestamp_Label` is the CSV row index in the retained CloudDDoS and Darknet
files. Its physical-time units are not established for the retained IIoTAPT
file. It must not be presented as verified elapsed seconds or calendar time.

## Provided row indices

`splits/ordered/<dataset>/` contains disjoint train, validation, and test indices
and their audits. Indices are zero-based source-row numbers, one integer per
line, without a header. Equal order keys stay in the same partition. Counts:

| Dataset | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| CloudDDoS | 84,592 | 28,198 | 28,197 |
| IIoTAPT | 47,894 | 15,965 | 15,965 |
| Darknet | 95,140 | 31,713 | 31,713 |

`splits/retained/` provides the original binary benchmark training/test indices;
these are a different protocol with overlap on CloudDDoS and Darknet.
`splits/limited_label/` contains test indices for the separate multiclass-schema
files. These index sets are not interchangeable.

`configs/data_sources.json` records the exact processed binary source hashes.
Re-exporting CSV values, changing their order, or drawing a fresh subset creates
a different source; regenerate splits for such a source.

## Conversion of a new flow CSV

`scripts/prepare_data.py` requires an explicit event-order choice, explicit
binary label mapping, and the feature-column list. It checks finite numeric
features, preserves feature order, stably orders events, registers endpoints in
encounter order, and creates disjoint splits and an audit. Scaling is deferred
to training and fitted on the training partition only. Endpoint string values
are not written to a mapping file by the conversion helper.

Use `--feature-columns /path/to/columns.json` to supply a different named feature
list with the configured width. This is a generic flow-CSV conversion utility;
the original packet-to-flow extraction, historical CloudDDoS sampling decisions,
and unavailable class-mapping history have not been reconstructed. The helper
does not automatically guess them or guarantee historical numerical results.

Source traffic download and reuse remain subject to each provider's conditions.
