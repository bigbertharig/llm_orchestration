# GPU Control Plane Leases

GPU allocation authority is stored in one locked state file:

```text
shared/gpus/leases.json
```

The implementation is `shared/agents/gpu_leases.py`. Single- and multi-GPU
claims are written atomically under `leases.json.lock`, so a split group is
either acquired completely or not acquired at all.

Do not write `leases.json` or worker `heartbeat.json` directly.

## Lease Contract

Each active lease records:

- `lease_id` and immutable `generation`;
- `controller`: `plans`, `benchmarks`, `supervision`, `interactive`, or
  `system`;
- `lane`, `owner`, and `run_id`;
- physical `gpu_ids`, ports, and runtime IDs;
- owner host/PID;
- acquisition, renewal, and expiry timestamps;
- optional controller metadata.

Expiry is not permission to steal a GPU. An expired lease continues blocking
new acquisition until an operator or reconciler proves that the owning process,
port, and runtime are no longer active, then releases it explicitly.

## Plan Behavior

GPU workers read the canonical lease store before claiming queued Plan work.
Any lease blocks Plan claims on its member GPUs and is published into the
worker heartbeat as:

- `leased`;
- `lease_controller`;
- `lease`;
- `busy`, `busy_lane`, `busy_owner`, and `busy_reason`;
- `available_for_lanes` and `blocked_by_lane`.

The brain excludes leased workers from model-capacity and split-placement
decisions. Plan-owned acquisition/renewal/release around active Plan work is the
next integration step; the current completed step is authoritative exclusion.

## Admission Policy

Operator intent is separate from GPU ownership:

```text
shared/brain/control_policy.json
```

`boot_profile` describes the desired startup posture. `admitted_controllers`
defines which controllers may create new work. Live activity is derived from
leases, runtimes, tasks, and heartbeats, never from the selected profile.

Profiles currently supported by `shared/agents/control_policy.py`:

| Profile | Default admitted controller |
| --- | --- |
| `neutral` | none |
| `plan_ready` | `plans` |
| `benchmark_ready` | `benchmarks` |
| `supervision_ready` | `supervision` |
| `custom` | `interactive` |

An explicit admitted-controller list may differ from the profile when the
operator wants non-conflicting mixed work.

## Operator Commands

Acquire GPUs 1 and 3 atomically for a Plan controller:

```bash
ssh gpu /home/bryan/llm-orchestration-venv/bin/python \
  /mnt/shared/scripts/manage_gpu_lease.py \
  --shared-path /mnt/shared acquire \
  --controller plans \
  --owner gpu-plan-controller \
  --run-id plan-pool-1 \
  --gpus 1,3 \
  --ports 11435,11437
```

Inspect leases:

```bash
ssh gpu /home/bryan/llm-orchestration-venv/bin/python \
  /mnt/shared/scripts/manage_gpu_lease.py \
  --shared-path /mnt/shared list
```

Renew an owned lease:

```bash
ssh gpu /home/bryan/llm-orchestration-venv/bin/python \
  /mnt/shared/scripts/manage_gpu_lease.py \
  --shared-path /mnt/shared renew \
  --lease-id LEASE_ID \
  --owner gpu-plan-controller
```

Release an owned lease:

```bash
ssh gpu /home/bryan/llm-orchestration-venv/bin/python \
  /mnt/shared/scripts/manage_gpu_lease.py \
  --shared-path /mnt/shared release \
  --lease-id LEASE_ID \
  --owner gpu-plan-controller
```

`--force-reconciled` deliberately requires omission of `--owner`. Use it only
after checking the owner process, GPU ports, runtime containers, and current
GPU utilization.

## Lane Display

Dashboard lane labels remain an operator display abstraction:

| Controller/lane | Display |
| --- | --- |
| benchmarks | `BENCHING` |
| plans | `TASKING` |
| interactive | `CHATTING` |
| supervision | `SUPERVISING` |

All four are mutually exclusive on a physical GPU in the initial control-plane
implementation.
