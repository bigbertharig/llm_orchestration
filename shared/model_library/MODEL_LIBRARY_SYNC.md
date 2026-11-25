# Orchestration model-library import

Benchmarking owns the identities, status, roles, measured profiles, recipes and evidence
inside its published artifact. Orchestration owns executable local runtime configuration.
Standalone registration of unbenchmarked models remains Planned; local configuration will
retain separate provenance rather than changing an imported snapshot.
Orchestration imports this knowledge into its own files and owns the executable
runtime settings and GPU placement. Rig chat services, benchmark campaigns, and model tools read the imported files,
not the benchmarking checkout.

On the Pi, after benchmarking exports and checks its model library:

```bash
cd ~/llm_orchestration
python3 shared/scripts/sync_model_library.py what-new
python3 shared/scripts/sync_model_library.py sync
python3 shared/scripts/sync_model_library.py check
```

`BENCHMARKING_ROOT` or `--source` selects a directory containing
`model_library_export.json`. The importer verifies the artifact schema/content hash and
model references using Orchestration code, then stages its outputs before replacing them.
It does not import or execute Benchmarking Python modules. `check`
returns 0 for current, 1 for pending, and 2 for invalid source. A missing
benchmarking checkout does not affect already imported runtime files.

| Orchestration-owned file | Runtime use |
| --- | --- |
| `model_library.json` | Imported registry, roles, and passive recipe records, with source hash. Rig chat services use it for reviewed candidate settings; model tools use it for paths and the runtime image. |
| `model_tuning_profiles.json` | Imported resolved tuning profiles with the source hash. |
| `models.catalog.json` | Local benchmark and runtime catalog, replacing the old symlink into Benchmarking. |
| `model_access.json` | Active interactive model IDs for local tools and runtime smoke checks. Candidates remain in the full snapshot for explicit tests but do not enter this automatic list. |

The sync changes files, not live processes. Existing services pick up the new
library on their next load or restart. It does not start or stop a model.
Experimental named model modes remain implemented by
`shared/scripts/rig/model_modes.py`; their canonical model IDs and GGUF paths
resolve from the imported snapshot. The mode's GPU map and launch flags remain
orchestration-owned code.
