# Human Action Required: Benchmark Thermal Shutdown

Date: 2026-08-20
Campaign: `modern_models_validation_202608/20260820_modern_validation_l50`

## What Happened

The modern-model validation campaign ran four concurrent model lanes:

- Qwen 3.5 27B on GPU 0
- Ministral 14B split across GPUs 1+3
- Ministral 8B split across GPUs 4+5
- Ministral 3B on GPU 2

The shared CPU temperature rose through 91-98C and reached the configured 99C
critical threshold. Worker agents shut down at 100C. The rig then rebooted at
approximately 14:35 PDT, terminating the campaign and all Docker containers.

Memory was not exhausted: earlyoom reported about 49-52% available RAM and
about 98% free swap immediately before the event.

## Required Checks

1. Inspect CPU cooler mounting, pump/fan operation, dust, airflow, and thermal paste.
2. Confirm the CPU temperature sensor used by all worker agents is the intended sensor.
3. Confirm PSU/circuit and chassis cooling are suitable for the observed roughly 930W aggregate GPU load.
4. After hardware clearance, rerun with `--max-active-models 2` initially and monitor CPU temperature continuously.
5. Do not resume `20260820_modern_validation_l50`; start a new run ID so the interrupted evidence remains distinct.

## Evidence

- `/mnt/shared/logs/startup-benchmark.log`
- `/mnt/shared/logs/benchmarks/campaigns/history/modern_models_validation_202608/20260820_modern_validation_l50/`
- `/mnt/shared/logs/brain_decisions.log`

The interrupted run must not be promoted to validated/full evidence.
