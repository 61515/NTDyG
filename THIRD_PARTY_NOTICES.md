# Third-party source

The temporal baseline adapters use a vendored subset of
[DyGLib](https://github.com/yule-BUAA/DyGLib), commit
`3aacc36b94b8d2d8293d70a74fdf6d39089b4163`.

The original MIT license is retained at `code/third_party/DyGLib/LICENSE`.
`UPSTREAM.json`, `CAWN_UPSTREAM.json`, and other upstream records in that
directory preserve file hashes and source URLs. The public package retains
the original notices. `code/cawn_position_encoder.py` supplies the local
parameter-free indexed position calculation used by the CAWN adapter.

This packaging does not assign a new project license or replace the licenses
of third-party code. `SHA256SUMS` records the included files' checksums.
