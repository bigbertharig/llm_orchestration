# Runtime experiments

Measurement drivers now belong to Benchmarking. Use its `scripts/experiments/runtime_campaign.py`
with explicit local profile variants and GPU placement. Register and load through `rig api`;
never invoke private runtime scripts from a campaign. The former expert-placement source is
preserved with hashes in Benchmarking `archive/experiment_sources_20261124/`.
Orchestration keeps runtime build, readiness and admission regression tests.
