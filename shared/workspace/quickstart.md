# Rig quickstart

The rig hardware layer accepts model work from Gateway and keeps the authoritative hardware state on the rig. The old Pi plan-submit and manual model-load wrappers have been removed.

## Inspect the rig

On the GPU host:

```bash
rig scan
rig observe
rig status
rig plan show
```

`rig scan` writes a timestamped hardware inventory. `rig observe` shows current GPUs, leases, claims, loaded services, guard, and active controller. An example inventory in the repository is not a requirement for another card mix.

## Preview capacity

```bash
rig lease preview --spec '{"gpus_needed":[{"min_vram_mb":8192}]}'
```

Preview is read-only and reports selected cards/port or a typed refusal. Capacity can change before an actual lease; admission checks again under the rig lock.

## Request a model

Use Gateway's Manager API or `rigctl` from the controller. Gateway selects a model and prepared service configuration; Orchestration performs physical admission, launches it, and returns observed readiness. The controller may run on the GPU host or another local machine. See Gateway's `docs/MANAGER.md` for request and lease commands.

## Results and legacy state

Benchmarking's deterministic campaign submitter sends grouped suite requests through Gateway. Benchmark receipts and model files may live on shared storage. The old brain/worker queues still use shared task and heartbeat files, and their status is available through the private Orchestration API. New work should not submit a plan file through the shared drive.

For rig service, guard, and recovery details see [the hardware layer guide](../scripts/rig/README.md). For the repository boundary see [CONTEXT.md](CONTEXT.md).
