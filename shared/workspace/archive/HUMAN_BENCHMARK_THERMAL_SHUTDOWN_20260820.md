# Resolved Hardware Check: Benchmark Thermal Shutdown

Date: 2026-08-20
Campaign: `modern_models_validation_202608/20260820_modern_validation_l50`

## What Happened

The modern-model validation campaign ran four concurrent model lanes:

- Qwen 3.5 27B on GPU 0
- Ministral 14B split across GPUs 1+3
- Ministral 8B split across GPUs 4+5
- Ministral 3B on GPU 2

The shared CPU package temperature rose through 91-98C and reached the
configured 99C critical threshold. Worker agents began shutting down at 100C.
The host then suffered an unclean hard reset: the previous boot's last journal
entry is 14:34:17 PDT and the new kernel boot began at 14:34:48 PDT. This
terminated the campaign and all Docker containers.

Memory was not exhausted: earlyoom reported about 49-52% available RAM and
about 98% free swap immediately before the event.

## Diagnostic Findings

- This was a real host reboot, not only a Docker or orchestration restart. The
  new boot recovered the root filesystem journal, cleared orphaned inodes, and
  reported that the prior journal was uncleanly shut down.
- There is no orderly shutdown, reboot command, kernel panic, OOM, MCE, NVIDIA
  Xid, or watchdog event in the journal before it ends abruptly.
- The worker safety path does not reboot the host. At the critical threshold it
  pauses workers, stops its own llama-server, and exits the GPU agent.
- The benchmark model servers were separate Docker containers, not child
  processes owned by the GPU agents. Each critical action reported `emergency
  paused 0 workers`; benchmark GPU load continued after agents exited. The
  current thermal guard therefore did not quiesce the workload that caused the
  incident.
- The reported temperature maps to Linux `x86_pkg_temp` / Intel `coretemp` on
  the i7-7700K. The raw `coretemp` package critical point is exactly 100C, so
  this was not an ACPI or storage-temperature misidentification.
- Current idle readings are about 37-43C. A February 2026 five-minute,
  eight-thread generic CPU stress baseline held about 75-76C. However,
  historical orchestration logs also contain multiple 96-100C LLM-load events
  in February, including worker critical shutdowns. The high CPU temperature is
  recurring; the unclean host reset is the new failure mode.
- Linux exposes ACPI fan cooling devices but no fan RPM or pump telemetry. Fan
  and pump operation cannot be verified remotely.
- The host remained alive for roughly two minutes after the first 100C worker
  shutdown while the independently managed benchmark containers continued.
  The final reset was also near the campaign's roughly 930W aggregate GPU-power
  peak. Software evidence therefore cannot distinguish a delayed CPU firmware
  thermal trip from PSU/circuit instability; both require physical checks.
- Motherboard: ASUS B250 MINING EXPERT, BIOS 1001 dated 2017-12-13. CPU turbo is
  enabled; the active Linux scaling governor is `powersave`.

## Required Checks

1. Inspect CPU cooler mounting, pump/fan operation, dust, airflow, and thermal paste.
2. Confirm CPU fan/pump operation in BIOS or by physical inspection; Linux has no usable RPM telemetry on this board.
3. Confirm PSU model/rating, rail distribution, cabling, wall circuit, and chassis cooling are suitable for the observed roughly 930W aggregate GPU load.
4. After hardware clearance, run a single controlled CPU thermal check and stop if it materially exceeds the prior 75-76C full-load baseline.
5. Before retrying, make the benchmark controller stop all campaign containers
   when the CPU reaches the critical threshold; worker-agent shutdown alone does
   not control leased benchmark containers.
6. Rerun with `--max-active-models 2` initially and monitor CPU package temperature and total GPU power continuously.
7. Do not resume `20260820_modern_validation_l50`; start a new run ID so the interrupted evidence remains distinct.

## Evidence

- `/mnt/shared/logs/startup-benchmark.log`
- `/mnt/shared/logs/benchmarks/campaigns/history/modern_models_validation_202608/20260820_modern_validation_l50/`
- `/mnt/shared/logs/brain_decisions.log`

The interrupted run must not be promoted to validated/full evidence.

## Controlled Retry

At approximately 20:03 PDT, the rig was cleanly rebooted over SSH after the
owner confirmed wiring and fan operation. The new boot ID is
`a8626138-dc2c-4f3b-97fd-bfc4724d9457`. The previous boot completed an orderly
systemd shutdown and the new boot reported no new hardware, MCE, NVIDIA Xid, or
filesystem-recovery errors.

Startup and campaign telemetry is being sampled every two seconds with
`/mnt/shared/scripts/startup_thermal_monitor.py`.

- Clean normal startup peaked at 68C CPU package temperature.
- A retry with two concurrent models averaged 84C over its final two minutes
  and reached 89-90C. It was stopped cleanly before reaching the critical limit.
- The active retry is `20260820_modern_validation_l50_retry3`, sequential with
  `--max-active-models 1`.
- Sequential Qwen 3.5 27B inference generally runs around 82-87C but has
  produced a brief 90C sample. GPU 0 runs near 450W and 71-74C.
- The active telemetry guard sends SIGTERM to the campaign runner at 95C so its
  independently managed benchmark containers are stopped before the 100C CPU
  hardware limit.

Active evidence:

- `/mnt/shared/logs/startup_thermal_clean_reboot_20260820_200248.csv`
- `/mnt/shared/logs/startup_thermal_modern_retry_20260820_200607.csv`
- `/mnt/shared/logs/startup_thermal_modern_retry_guarded_20260820_201429.csv`
- `/mnt/shared/logs/startup_thermal_modern_retry_guarded_20260820_201429.events.log`
- `/mnt/shared/logs/benchmarks/campaign-20260820_modern_validation_l50_retry3.log`

## Closure

The owner checked the wiring and all fans, and the clean-reboot diagnostics did
not identify a hardware or boot fault. Sequential retry
`20260820_modern_validation_l50_retry3` completed all 12 manifest lanes without
activating the 95C guard. Across the guarded run, CPU package temperature peaked
at 94C, with 16 samples at or above 90C and none at or above 95C; GPU temperature
peaked at 79C. This clears the physical-check action, but sequential execution
and the 95C campaign guard remain required on this host.

The post-run evidence audit found a separate benchmark-controller defect. The
three brain lanes reused an existing `qwen3.6:27b` server on port 11434 while
attributing results to Qwen3.5, Qwen Coder, and Devstral. Those brain records are
invalid and were removed from the canonical ledger. The Ministral 14B/8B
reasoning/runtime and Ministral 3B code/runtime results used the requested model
runtimes and remain valid. Brain-only validation must wait for endpoint identity
enforcement; this is tracked in
`workspace/implement/benchmark_runtime_identity_preflight.md`.
