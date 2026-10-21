# Processed dataset archives

This directory contains four gzip files: the binary and limited-label schemas
for IIoTAPT and Darknet. These are the retained, processed event tables, not the
original packet captures. Every archive is below GitHub's 25 MiB browser-upload
limit. CSV filenames, byte counts, and SHA-256 values are recorded in
[`configs/processed_data_archives.json`](../../configs/processed_data_archives.json).

From the release root, run:

```bash
python scripts/unpack_data.py
```

This writes the original CSV bytes to `data/binary/` and `data/multiclass/`,
checks each archive and CSV checksum, and preserves existing files. An existing
CSV with a different checksum causes an error. Use `--schema binary` to unpack
only binary data. Expanded CSVs remain excluded by `.gitignore`; upload the
compressed archives with the source files.

CloudDDoS is not bundled: redistribution permission was not established from
the BCCC provider pages inspected. Its source links, schema, and retained CSV
hashes are still provided. See [dataset attribution](../../docs/DATA_ATTRIBUTION.md).
