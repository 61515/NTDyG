Optional limited-label experiments use separate processed files named
`CloudDDoS_re.csv`, `IIoTAPT_re.csv`, and `Darknet_re.csv`. These have the retained
multiclass schema: numeric flow features plus `Timestamp_Label`, `Src IP Label`,
`Dst IP Label`, `Label1_encoded`, and `Label2_encoded`.
`Label2_encoded` supplies 26, 2, and 8 classes, respectively.
These are different files from the binary-schema datasets.

Run `python scripts/unpack_data.py --schema multiclass` from the release root
to unpack the included IIoTAPT and Darknet tables here. CloudDDoS must be
obtained separately. Sources and citations are in `docs/DATA_ATTRIBUTION.md`.
