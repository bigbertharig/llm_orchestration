# Orchestration model-library import

Benchmarking owns model IDs, status, roles, profiles, recipes, and evidence.
Orchestration imports this knowledge into its own files and owns the executable
runtime settings and GPU placement. Ordinary brain, worker, rig, and model-access
processes read the imported files, not the benchmarking checkout.

On the Pi, after benchmarking exports and checks its model library:

```bash
cd ~/llm_orchestration
python3 shared/scripts/sync_model_library.py what-new
python3 shared/scripts/sync_model_library.py sync
python3 shared/scripts/sync_model_library.py check
```

`BENCHMARKING_ROOT` or `--source` selects another checkout. The importer runs
benchmarking's existing `model_library.py check` during sync, verifies role and
recipe model references, and stages its outputs before replacing them. `check`
returns 0 for current, 1 for pending, and 2 for invalid source. A missing
benchmarking checkout does not affect already imported runtime files.

| Orchestration-owned file | Runtime use |
| --- | --- |
| `model_library.json` | Imported registry, roles, and passive recipe records, with source hash. Brain and workers resolve their models and llama settings from it. Rig tools use it for model paths and the default runtime image. |
| `models.catalog.json` | Local scheduling and model-access catalog, replacing the old symlink into benchmarking. |
| `model_access.json` | Active interactive model IDs for the orchestration model-access API. Candidates remain in the full snapshot for explicit tests but do not enter this automatic list. |

The sync changes files, not live processes. Existing services pick up the new
library on their next load or restart. It does not start or stop a model.
Experimental named model modes remain implemented by
`shared/scripts/rig/model_modes.py`; their canonical model IDs and GGUF paths
resolve from the imported snapshot. The mode's GPU map and launch flags remain
orchestration-owned code.
