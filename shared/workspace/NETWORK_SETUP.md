# Network Setup

Reference for the physical network, shared-drive layout, and host-to-host trust
assumptions behind the orchestration rig.

This is an infrastructure reference, not the normal operator runbook. For daily
operations, use [quickstart.md](quickstart.md) and [CONTEXT.md](CONTEXT.md).

---

## Topology

```text
Internet
  -> home router
  -> dedicated rig router (10.0.0.1)
  -> switch(es)
     -> control plane / Raspberry Pi (10.0.0.2)
     -> GPU rig (10.0.0.3)
     -> optional CPU workers (10.0.0.10+)
```

The rig is local-network-first. Shared storage and local SSH are the normal
control channels.

---

## Addressing

| Device | Address | Notes |
|--------|---------|-------|
| Router | `10.0.0.1` | dedicated rig router |
| Control plane | `10.0.0.2` | repo checkout, shared drive, operator entrypoints |
| GPU rig | `10.0.0.3` | brain + GPU workers |
| CPU workers | `10.0.0.10+` | optional worker pool |

DHCP pool is on the `10.0.0.0/24` subnet.

---

## Trust Boundary

Important assumptions:

- the LAN is trusted relative to the internet
- physical access to the LAN is powerful access
- the router is the perimeter
- no inbound public port-forwarding should exist for orchestrator control

This means local services can trust the rig LAN more than the public internet,
but not more than the human operator.

---

## Shared Drive Layout

Authoritative shared drive:

- `/media/bryan/shared`

Repo-side bind mount:

- `/home/bryan/llm_orchestration/shared`

GPU-rig NFS mount:

- `/mnt/shared`

This shared drive is the common substrate for:

- task lanes
- worker heartbeats
- batch history
- shared scripts
- models archive

---

## Mount Model

### Control Plane

The control plane mounts the physical disk at:

- `/media/bryan/shared`

and bind-mounts it into the repo at:

- `/home/bryan/llm_orchestration/shared`

### GPU Rig

The GPU rig mounts the shared drive over NFS at:

- `/mnt/shared`

This is the normal path used by the orchestrator running on the GPU host.

---

## NFS

Server:

- control plane / Pi

Client:

- GPU rig
- optional CPU workers

Key property:

- orchestration communication is largely file-based over the shared mount

If NFS is down, orchestration is not in a safe normal state.

---

## SSH

Normal SSH trust paths:

- control plane -> GPU rig
- GPU rig -> control plane

This is used for:

- submit wrapper behavior when targeting the rig
- startup/reset helper scripts
- maintenance commands

SSH here is an operator/control-plane tool, not the main runtime data plane.

---

## Runtime Ownership Note

This doc should not be read as “the rig is driven by fixed always-on Ollama
services.”

Current normal runtime model is:

- startup scripts bring up the orchestrator
- brain owns shared coordination
- workers are cold by default unless startup mode says otherwise
- worker model load/unload happens through orchestrator `meta` tasks or explicit
  startup-mode wrappers

Older host-specific Ollama service layouts are historical infrastructure notes,
not the primary operator workflow.

Relevant startup entrypoints:

- `/home/bryan/llm_orchestration/shared/agents/startup.py`
- `/home/bryan/llm_orchestration/scripts/start_default_mode.py`
- `/home/bryan/llm_orchestration/scripts/start_benchmark_mode.py`
- `/home/bryan/llm_orchestration/scripts/start_custom_mode.py`

---

## Models And Storage

Shared archive storage:

- `/media/bryan/shared/models/`

The shared drive is the archive/source of model assets. Local host runtime
storage is the hot set.

This keeps model promotion and eviction explicit instead of hidden inside the
network layout.

---

## Troubleshooting Anchors

When the network/storage layer is suspect, check:

1. router reachability
2. control-plane reachability
3. NFS mount state on the GPU rig
4. shared-drive visibility under `/mnt/shared`
5. SSH connectivity between control plane and GPU rig

Useful checks:

```bash
ping 10.0.0.2
ping 10.0.0.3
mount | grep /mnt/shared
nc -zw2 10.0.0.2 2049
ssh gpu hostname
ssh pi hostname
```

If the shared mount is unavailable, stop treating task lanes and history
artifacts as trustworthy until the mount is healthy again.

---

## Historical Notes Worth Keeping

These topics still matter historically, but they are not the main operator
story anymore:

- host-specific dual-Ollama service experiments
- old fixed worker/brain service pinning
- desktop convenience shortcuts on the GPU rig

If those become operationally important again, document them as host-specific
appendices instead of the primary network model.
