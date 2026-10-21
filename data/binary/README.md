Run `python scripts/unpack_data.py --schema binary` from the release root to
place the included processed `IIoTAPT_re.csv` and `Darknet_re.csv` here.
Obtain `CloudDDoS_re.csv` separately from its provider. Each file has numeric flow features followed by
`Timestamp_Label`, `Label_encoded`, `Src IP Label`, and `Dst IP Label`.
Use `scripts/prepare_data.py` for a new, explicitly mapped source CSV.
The exact retained row-index files require the source checksums in
`configs/data_sources.json`; regenerated CSVs require freshly generated splits.
Dataset sources and required citations are in `docs/DATA_ATTRIBUTION.md`.
