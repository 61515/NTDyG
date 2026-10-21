Optional repetitions of the retained fixed-memory classifier require the
original memory inputs for each dataset in its own subdirectory:

- `trained_GRU_memory_model.pt`
- `memory_updates_each_event.csv`
- `final_memory_matrix.csv`
- `test_matrix.csv`

The main validation-selected runner trains its own memories and does not need
these files. The original fixed-memory inputs are not bundled with this release;
freshly trained memory weights do not reconstruct the fixed inputs used by the
paper's main benchmark repetitions.
