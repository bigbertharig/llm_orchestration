# LLM Orchestration — Quick Reference

Multi-GPU LLM orchestration: Claude writes plans, Brain interprets and sequences, GPU agents execute tasks via workers.

## Where To Find Things

| Topic | Read This |
|-------|-----------|
| Architecture & hardware | `workspace/architecture.md` |
| Plan format spec | `workspace/PLAN_FORMAT.md` |
| Quick start (status, submit, monitor) | `workspace/quickstart.md` |
| Brain loop & task handling | `workspace/brain-behavior.md` |
| End-to-end workflow | `workspace/distributed_work_guide.md` |
| Network setup (NFS, SSH, static IPs) | `workspace/NETWORK_SETUP.md` |
| Hardware config & GPU assignment | `agents/config.json`, `agents/setup.py` |
| Security rules | `core/RULES.md` |
| Benchmarking guide | `workspace/llm_benchmark_testing_guide.md` |
| Priority tasks (do first) | `workspace/implement/` |
| Dashboard (web UI) | `scripts/dashboard.py` — run via `python scripts/dashboard.py`, access at http://localhost:8787 |
| Cloud escalation artifact schema | `workspace/escalation_artifact_schema.md` |
| Plan organization (shoulders & arms) | `plans/README_SHOULDERS_ARMS.md` |

## File Layout

```
shared/
├── agents/        # brain.py, gpu.py, worker.py, executor.py, setup.py, startup.py
├── core/          # PROTECTED (root-owned) — system prompt, rules, escalation
├── workspace/     # Docs, implement/, future/, human/, archive/
├── plans/         # Plan folders (each its own git repo, gitignored from main)
├── tasks/         # Runtime queue (queue/, processing/, complete/, failed/)
├── brain/         # Brain state, private_tasks/, escalations/ (runtime)
├── gpus/          # GPU heartbeats (runtime)
├── signals/       # GPU control signals (runtime)
└── logs/          # Logs
```

## Key Principles

- No backwards compatibility — one clean way, delete old patterns
- Fail fast with clear errors
- shared/ lives on external drive, bind-mounted into repo, NFS-shared to GPU rig
- Plans are independent git repos (shared/plans/ is gitignored from main repo)
- Priority tasks go in workspace/implement/, completed work archived to workspace/archive/
