# Multi-Agent GPU Cluster Systems Analysis Questionnaire

Use this document as a structured review of the current orchestration system.
It is not a speculative design doc. Each section should capture:

1. what the system does today
2. what the source of truth is
3. what is still unresolved

For operator commands, use [quickstart.md](quickstart.md). For architecture and
brain ownership rules, use [architecture.md](architecture.md) and
[brain-behavior.md](brain-behavior.md).

---

## 1. System Architecture

### 1.1 Component Inventory

- [x] What hardware is in the rig?
- [x] What software stack is active on each major component?
- [x] What are the key resource constraints?

**Current answers**

- Control plane:
  - Raspberry Pi hosts the repo checkout, shared drive bind mount, and operator entrypoints.
- GPU rig:
  - NVIDIA multi-GPU host runs the brain and GPU workers.
  - Worker inventory and placement are defined by `shared/agents/config.json`.
- Shared storage:
  - `/media/bryan/shared` is the authoritative shared filesystem.
  - `/home/bryan/llm_orchestration/shared` is the repo-side bind mount.
- Runtime stack:
  - `startup.py` starts the orchestrator.
  - `brain.py` owns plan interpretation and shared coordination.
  - `gpu.py` owns per-GPU worker control loops.
  - optional CPU workers use the same shared task lanes.

**Open questions**

- [ ] Do we want a compact hardware inventory doc generated from `config.json` and `nvidia-smi` instead of maintaining hardware prose by hand?

### 1.2 Communication Topology

- [x] How do components communicate?
- [x] What is the source of truth for runtime state?
- [x] What happens if the communication layer fails?

**Current answers**

- Communication is primarily file-based through the shared filesystem.
- Core runtime state lives in:
  - `shared/tasks/{queue,processing,complete,failed}/`
  - `shared/brain/`
  - `shared/gpus/` and `shared/cpus/`
  - `shared/plans/<plan>/history/<batch_id>/`
- The brain is the source of truth for shared coordination state:
  - queue truth
  - task release and requeue
  - split cleanup and quarantine
  - global shared-runtime ownership decisions
- Workers are the source of truth for local observations:
  - local process state
  - local telemetry
  - local execution results

**Failure mode**

- If the shared filesystem is unavailable, orchestration cannot safely continue.
- Safe behavior is to stop making new shared-state decisions and escalate.

### 1.3 Role Definitions

- [x] What is each agent's responsibility?
- [x] What decisions can each tier make autonomously?
- [x] What requires escalation?

**Current answers**

- Brain:
  - parses plans
  - creates task graphs
  - releases work
  - handles retries, requeues, and shared cleanup decisions
  - refreshes per-run summary artifacts
- Workers:
  - claim eligible tasks
  - execute local work
  - handle local probes and child-process cleanup
  - report observations upward
- Dashboard:
  - operator UI over the same orchestration system
  - normal controls: `Start Plan`, `Kill Plan`, `Reset selected GPU`, `Return To Default`
- Human:
  - final authority for protected files, security-sensitive changes, and unresolved escalations

**Authority rule**

- If a decision affects only local worker state and is provably isolated, the worker may own it.
- If a decision affects shared state, scheduling, queue truth, or another worker, the brain must own it.

---

## 2. Task Management

### 2.1 Task Lifecycle

- [x] How is a task created, assigned, executed, and marked terminal?
- [x] What states can a task be in?
- [x] Where is task state stored?

**Current answers**

1. Plan submission writes an `execute_plan` task.
2. The brain parses `plan.md` and creates private tasks in `shared/brain/private_tasks/`.
3. Tasks release into `shared/tasks/queue/` when dependencies are met.
4. Eligible workers claim tasks into `shared/tasks/processing/`.
5. Tasks end in `shared/tasks/complete/` or `shared/tasks/failed/`.
6. Brain-owned requeue paths normalize stale runtime fields before returning work to `queue/`.

**Current task states**

- `pending`
- `processing`
- `complete`
- `failed`
- `abandoned` is a terminal batch/task outcome used in lifecycle handling even if not represented as a separate lane

### 2.2 Task Schema

- [x] What fields define a task?
- [x] How are dependencies enforced?
- [x] How do task classes differ?

**Current answers**

Minimum common fields:

```json
{
  "task_id": "uuid",
  "batch_id": "abc123",
  "name": "prepare_repo",
  "type": "shell",
  "command": "python3 ...",
  "executor": "brain | worker",
  "task_class": "cpu | script | llm | brain | meta",
  "depends_on": ["other_task"],
  "priority": 5,
  "status": "pending",
  "retry_count": 0
}
```

Task class meaning:

- `cpu`: CPU-only work
- `script`: non-LLM scripted work, may still use GPU
- `llm`: model-backed worker work
- `brain`: brain-owned logic
- `meta`: runtime control tasks like load/unload/reset

### 2.3 Prioritization

- [x] How does the scheduler decide what runs next?
- [ ] Do high-priority tasks preempt running work?
- [ ] How is starvation prevented?

**Current answers**

- Brain controls release order.
- Workers claim from the public queue based on eligibility, class, and priority.
- `meta` tasks are highest priority because they control runtime state.
- There is no true preemption of already-running worker subprocesses.

**Open questions**

- [ ] Do we need explicit fairness/starvation protection, or is current queue behavior sufficient for the workloads we actually run?

---

## 3. Resource Management

### 3.1 Runtime Ownership

- [x] Who decides when models load and unload?
- [x] Who decides split-runtime cleanup?
- [x] Who decides quarantine and lease reclaim?

**Current answers**

- Brain owns shared runtime authority.
- Workers detect and report conditions.
- Workers execute fenced brain-issued runtime commands.
- Split cleanup, split quarantine, reservation terminal handling, and stale shared-owner reclaim are brain-decided paths.

### 3.2 GPU Allocation

- [x] How does the scheduler know GPU health and availability?
- [x] How are worker runtimes switched between states?
- [x] What does the worker manage locally?

**Current answers**

- Worker heartbeats report GPU health, loaded runtime state, and issue observations.
- Brain consumes worker heartbeats and decides shared runtime actions.
- Workers still manage:
  - subprocess supervision
  - local VRAM/temperature/port probes
  - local child cleanup

### 3.3 Thermal and Power

- [x] Are temperatures and VRAM monitored?
- [x] Is there recovery logic when a GPU overheats or wedges?
- [x] What is the operator-facing reset path?

**Current answers**

- GPU telemetry is monitored through worker heartbeats and `nvidia-smi`.
- Thermal recovery exists in agent/runtime logic.
- Normal operator reset path is the dashboard:
  - `Reset selected GPU` for one worker
  - `Return To Default` for full rig normalization

**Open questions**

- [ ] Do we want stronger policy around automatic thermal backoff versus manual operator intervention?

---

## 4. Error Handling and Recovery

### 4.1 Failure Modes

- [x] What happens if a worker crashes mid-task?
- [x] What happens if the brain aborts a batch?
- [x] What happens to partial runs?

**Current answers**

- The brain detects stale worker state and can requeue or abandon impacted tasks through centralized helpers.
- Fatal batch aborts still write failure artifacts under the batch history folder.
- Per-run summary artifacts are refreshed from brain lifecycle code, not only from a terminal summary task.
- Partial runs can still be reviewed afterward with the history summary reducers.

### 4.2 Retry Logic

- [x] How many times are failed tasks retried?
- [x] Who decides retries?
- [ ] Is there adaptive backoff?

**Current answers**

- Brain owns retry and requeue policy.
- Failed tasks may be retried according to brain failure handling.
- Requeue scrubbing is centralized so stale `assigned_to`, `started_at`, and result fields do not leak back into `queue/`.

**Open questions**

- [ ] Should retry policy become explicitly data-driven per plan/task class instead of mostly core-brain policy?

### 4.3 State Recovery

- [x] Can a run be inspected after interruption?
- [x] Are per-run events durable?
- [x] Are summary artifacts recoverable offline?

**Current answers**

- Brain writes:
  - `logs/batch_events.jsonl`
  - `RUN_SUMMARY.json`
  - `RUN_SUMMARY.md`
- Offline tools can re-summarize a single run or roll up many runs even if the batch ended in failure or partial state.

---

## 5. Monitoring and Observability

### 5.1 Metrics Collection

- [x] What metrics and artifacts are logged?
- [x] Where are they stored?
- [x] What review surfaces exist?

**Current answers**

- Live runtime:
  - `shared/logs/brain_decisions.log`
  - worker heartbeats under `shared/gpus/` and `shared/cpus/`
  - task lanes under `shared/tasks/`
- Per-run:
  - `history/<batch_id>/logs/batch_events.jsonl`
  - `history/<batch_id>/RUN_SUMMARY.*`
  - plan-specific artifacts under `output/` and `results/`
- Cross-run:
  - `history/_summary/` via the rollup script

### 5.2 Alerting

- [x] What operator-facing monitoring exists?
- [ ] What alert thresholds are formalized?
- [ ] How should alerts be delivered?

**Current answers**

- Dashboard exists and shows:
  - active alerts
  - brain GPU status
  - worker state
  - active batches
  - task lanes
- The main live log is `shared/logs/brain_decisions.log`.

**Open questions**

- [ ] Do we want push-style alerts, or is dashboard + log watching enough for now?

### 5.3 Review Tools

- [x] How do we reduce context before LLM review?
- [x] Is there a one-run reducer?
- [x] Is there a many-run reducer?

**Current answers**

- One run:
  - `scripts/summarize_history_run.py`
- Many runs:
  - `scripts/rollup_history.py`
- These tools reduce raw history trees into smaller review surfaces. They do not make decisions by themselves.

---

## 6. Escalation Path

### 6.1 Local to Human Escalation

- [x] When should the system stop and escalate?
- [x] Where is escalation documented?

**Current answers**

- Protected and unresolved issues escalate through:
  - `shared/brain/escalations/` for structured cloud/local escalation artifacts
  - `workspace/human/HUMAN_{topic}.md` for human review
- Immutable escalation and safety rules live in `shared/core/`.

### 6.2 External / Cloud Tier

- [ ] Is the cloud escalation path a production dependency?
- [ ] What context is handed off?
- [ ] How are responses reintegrated?

**Current answers**

- The local system should remain operational without a cloud tier.
- Cloud escalation is additive, not required for normal local orchestration.
- If used, it should consume structured escalation artifacts, not bypass the brain.

**Open questions**

- [ ] Do we want to formalize cloud escalation as a real subsystem, or keep it as an operator-assisted path?

---

## 7. Security and Isolation

### 7.1 Protected Areas

- [x] What is immutable?
- [x] What is agent-writable?
- [x] What should never be touched automatically?

**Current answers**

- Immutable/protected:
  - `shared/core/`
  - permission policy files
  - secrets and credential stores
- Agent-writable working areas:
  - `shared/workspace/`
  - plan repos under `shared/plans/`
  - runtime lanes and history artifacts

### 7.2 Blast Radius

- [x] What can a worker damage?
- [x] What can the brain damage?
- [x] How is that constrained?

**Current answers**

- Workers and brain may write normal shared workspace/runtime artifacts.
- They should not modify protected core instruction files.
- Shared authority is centralized in the brain to reduce cross-worker races.
- Operator-facing commands route through the orchestrator or dashboard, not ad hoc direct runtime manipulation.

### 7.3 Submission Trust Boundary

- [x] Who submits plans?
- [x] What is the normal front door?
- [x] What is debug-only?

**Current answers**

- Normal front doors:
  - `scripts/submit.py`
  - dashboard `Start Plan`
- Direct agent submit paths are debug-only and should not be treated as the normal operator workflow.

---

## 8. Data, Learning, and Continuous Improvement

### 8.1 What We Keep

- [x] Do we keep both successes and failures?
- [x] Do we keep per-run summaries?
- [x] Do we keep cross-run rollups?

**Current answers**

- Yes:
  - task outcomes
  - per-run event logs
  - per-run summaries
  - cross-run rollups

### 8.2 How We Use It

- [x] Can we inspect one batch quickly?
- [x] Can we inspect many batches for trends?
- [ ] Are lessons normalized enough for automated comparison?

**Current answers**

- Single-run review is good enough for fast operator/LLM inspection.
- Cross-run review now exists via rollups.

**Open questions**

- [ ] Do we want a stronger normalized failure-signature or lesson schema, or is the current reducer output enough?

---

## 9. Current Gaps Worth Re-Checking Periodically

- [ ] fairness/starvation policy in task release/claiming
- [ ] stronger adaptive retry/backoff policy
- [ ] formal shared-filesystem failure handling
- [ ] clearer cloud-escalation product decision
- [ ] normalized lesson/failure taxonomy for cross-run analysis

This questionnaire should be updated when the answers change, not when we just
wish they were different.
