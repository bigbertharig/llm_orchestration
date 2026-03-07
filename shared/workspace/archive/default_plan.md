# Split GPU Readiness State Machine Fix

## Context

Workers can treat split runtime as ready before readiness is fully verified, then claim-time attestation sees `actual=[]`, causing reset storms. This is a state-consistency bug, not just a timing issue.

**Root Causes:**
1. Both launcher AND follower can promote reservation to "ready" via the `runtime_already_loaded` path
2. No stability gate - promotion happens immediately when port has model
3. No ready token mechanism to verify readiness
4. Port-based cleanup is unreliable when process group isn't tracked
5. Reachable port without target model can still infer readiness in brain

---

## Implementation Status

| Part | Status | Notes |
|------|--------|-------|
| **A: Readiness State Machine + Token** | COMPLETE | Token verification now fails closed on any missing/mismatched token |
| **B: Process Group Cleanup** | COMPLETE | `_stop_split_runtime()` now uses owner metadata first for reliable cleanup |
| **C: Auto-Return to Default Policy** | COMPLETE | Freshness checks applied consistently across all auto-default state decisions |

---

## Implementation Summary

### Part A: Readiness State Machine + Token — COMPLETE

**Fully implemented.** Token verification now fails closed:
- If reservation has `ready_token`, owner meta MUST have matching token
- Missing owner token = fail
- Stale owner token without reservation token = fail
- See `/home/bryan/llm_orchestration/shared/agents/gpu_split.py:1243-1257`

#### 1. Constants (`gpu_constants.py`) — DONE

Added at `gpu_constants.py:109-143`:

```python
# Stability gate configuration
READY_STABLE_PROBE_COUNT = 3          # Consecutive /api/ps hits required
READY_STABLE_PROBE_INTERVAL_SECONDS = 2
READY_MIN_AGE_SECONDS = 10            # Minimum token age before claiming

# Phased mismatch recovery
ATTESTATION_MISS_SOFT_FAIL_THRESHOLD = 1
ATTESTATION_MISS_HARD_FAIL_THRESHOLD = 2

# New reservation statuses
SPLIT_RESERVATION_STATUSES = {
    "waiting_partner", "joining", "loading",
    "warming",           # NEW: Warmup in progress
    "ready_stabilizing", # NEW: Post-warmup stability gate
    "ready", "failed", "unloading", "unloaded", "expired"
}

SPLIT_RESERVATION_LOADING_STATES = {
    "waiting_partner", "joining", "loading", "warming", "ready_stabilizing"
}
```

#### 2. Reservation Schema Changes (`gpu_split.py`) — DONE

New fields in reservation dict (used throughout `gpu_split.py`):
- `ready_token` - UUID issued when transitioning to ready
- `ready_token_issued_at` - ISO timestamp
- `stable_probe_count` - Current consecutive successful probes
- `warmup_completed_at` - When warmup phase completed
- `pending_attestation_miss_count` - `{worker_name: miss_count}` for phased recovery

#### 3. Stability Gate Implementation (`gpu_split.py`) — DONE

Implemented at `gpu_split.py:1305-1426`:

`_issue_ready_token()` at line 1305:
- Generate UUID token
- Record `ready_token_issued_at` timestamp
- Only callable by launcher
- Also updates owner meta file

`_run_stability_gate()` at line 1356:
- Only launcher runs the gate
- Poll `/api/ps` N consecutive times
- Require target model in every consecutive probe
- On pass: issue ready_token UUID and transition to "ready"

#### 4. Fix `runtime_already_loaded` Path — DONE

Fixed at `gpu_split.py:2541-2600`:
- Check if caller is launcher before any promotion (line 2554)
- If launcher + token exists + valid age: allow promotion (lines 2556-2567)
- If launcher + no token: transition to `ready_stabilizing` and run gate (lines 2568-2594)
- If NOT launcher: skip promotion entirely (lines 2595-2600)

Also fixed post-warmup path at `gpu_split.py:2843-2876`:
- After `_start_split_runtime` succeeds, set status to `"ready_stabilizing"` (not `"ready"`)
- Record `warmup_completed_at` timestamp
- Do NOT call `_set_split_runtime_loaded` — deferred until stability gate passes

#### 5. Fix Follower Mirroring — DONE

Fixed at `gpu_split.py:2704-2767`:
- Before mirroring ready state, verify `ready_token` exists and is valid (lines 2716-2732)
- Verify token age >= `READY_MIN_AGE_SECONDS` via `_verify_ready_token_age()`
- Follower NEVER calls `_set_reservation_status(..., "ready", ...)` — only mirrors via `_set_split_runtime_loaded`
- Follower only mirrors when launcher has already promoted and token is valid

#### 6. Claim Validation Changes (`gpu_tasks.py`) — DONE

Implemented at `gpu_tasks.py:174-284`:

`_validate_split_ready_token()`:
- Check reservation status == "ready"
- Check ready_token exists
- Check `now - ready_token_issued_at >= READY_MIN_AGE_SECONDS`
- Direct attestation on split port shows target model

Called before claiming split-tier LLM tasks.

#### 7. Phased Mismatch Recovery (`gpu_tasks.py`) — DONE

Implemented at `gpu_tasks.py:286-328`:

`_handle_attestation_miss()`:
- First miss: soft-fail + requeue, keep runtime (no reset)
- Second consecutive miss (>= `ATTESTATION_MISS_HARD_FAIL_THRESHOLD`): mark failed + cleanup
- Track miss count per worker in local state (`pending_attestation_miss_count`)
- `_clear_attestation_miss_count()` resets after successful claim

This prevents reset storms from transient misses.

#### 8. Brain Reconciliation Fix (`brain_resources.py`) — DONE

Implemented at `brain_resources.py:611-777`:

In `_reconcile_split_group_state()`:
- Check for valid `ready_token` before classifying as `ready_real` (lines 633-635)
- If port has model but NO ready_token: classify as `loading_suspicious` (lines 641-663)
- Only classify as `ready_real` if has valid token (lines 665-713)
- Reachable port without target model treated as `wedged_port` (lines 715-747)

#### 9. Runtime Owner File Enhancement (`gpu_split.py`) — DONE

Implemented at `gpu_split.py:1166-1249`:

`_write_split_runtime_owner_meta()` at line 1166:
- Includes `pgid` field for process group cleanup
- Includes `ready_token` field for readiness verification

`_update_split_runtime_owner_meta_token()` at line 1204:
- Updates owner meta with ready_token after stability gate passes

`_verify_owner_meta_matches_reservation()` at line 1216:
- Compare launcher, model_id, ready_token between owner meta and reservation
- Mismatch = `failed_precondition`

---

### Part B: Process Group Cleanup — COMPLETE

**Fully implemented.** `_stop_split_runtime()` now uses owner metadata first:
1. Reads owner meta to get group_id
2. Calls `_force_kill_split_runtime_owner(group_id)` first (authoritative path)
3. Falls back to Popen handle as secondary cleanup
4. Can fully clean leaked split runtimes even if parent `ollama serve` already exited
- See `/home/bryan/llm_orchestration/shared/agents/gpu_split.py:1767-1822`

#### 1. Launch Split Runtime with Process Group — DONE

Implemented at `gpu_split.py:1518-1540`:
- `start_new_session=True` in Popen (line 1523)
- Captures `pgid = os.getpgid(proc.pid)` (lines 1528-1532)
- Passes pgid to `_write_split_runtime_owner_meta()` (line 1539)

#### 2. Extend Owner Metadata Schema — DONE

Implemented at `gpu_split.py:1166-1193`:
- `pgid` field in payload (line 1186)
- Used for process group cleanup

#### 3. Add Authoritative Kill-by-Owner Helpers — DONE

Implemented at `gpu_split.py:2100-2240`:

`_kill_process_group()` at line 2100:
- SIGTERM first, wait up to 3s, then SIGKILL if needed
- Returns detailed result dict

`_kill_pid()` at line 2158:
- Direct SIGKILL, returns True if killed or already dead

`_force_kill_split_runtime_owner()` at line 2183:
- Reads owner meta, kills by PGID first, then PID fallback

#### 4. Wire Helper into Coordinated Cleanup — DONE

Implemented at `gpu_split.py:421-425, 451`:
- Calls `_force_kill_split_runtime_owner(group_id)` FIRST (before port cleanup)
- Includes `owner_kill` result in return dict

#### 5. Harden `_stop_split_runtime()` — DONE

Implemented at `gpu_split.py:1767-1805`:
1. Gets PGID from `self.split_runtime_process.pid`
2. Calls `_kill_process_group(pgid)` if available
3. Fallback to terminate/kill on process object
4. Clears `split_runtime_process = None`
5. Calls `_close_split_runtime_log_file()`
6. Calls `_clear_split_runtime_owner_meta()`

#### 6. Add Post-Cleanup Verification Gate — DONE

Implemented at `gpu_split.py:478-489`:
- Calls `_verify_no_ollama_runners_on_port(split_port)`
- If leaked runners found: logs `SPLIT_CLEANUP_LEAK`
- Retries with `_kill_local_listener_on_port()` and `_kill_orphan_ollama_runners()`
- Sets `result["cleanup_leak_retry"] = True`

---

### Part C: Auto-Return to Default Policy — COMPLETE

When batches are finished and task lists are clean, automatically normalize to default state (brain + worker on gpu-2).

---

#### What's Working

| Component | Location | Status |
|-----------|----------|--------|
| State fields in brain.py | lines 203-213 | DONE |
| `_is_system_globally_idle()` | `brain_resources.py:1476-1530` | DONE |
| `_is_fresh_gpu_state()` | `brain_resources.py:1402-1420` | DONE |
| `_has_real_work_in_flight()` | `brain_resources.py:1422-1475` | DONE |
| `_check_auto_default_policy()` | `brain_resources.py:1532-1700+` | DONE |
| Phased state machine (`None`→`normalizing`→`default_ready`) | lines 1570-1700+ | DONE |
| Split unload deduplication by group_id | with freshness checks | DONE |
| Separate Phase A (unloads) and Phase B (load default) | lines 1570-1620 | DONE |
| `actions: []` instead of lossy single `action` | lines 1548-1551 | DONE |
| Freshness gating for all state classification | multiple locations | DONE |

---

#### Fixed Bugs (5 Issues) — ALL RESOLVED

**Bug 1: Phase resets from auto-default's own meta tasks** — FIXED
- Was: `brain_resources.py:1493-1498` unconditionally reset phase
- Fix: Lines 1563-1600 now use `_has_real_work_in_flight()` to distinguish

**Bug 2: Idle timer restarts from auto-default's own work** — FIXED
- Was: `auto_default_last_busy_at = now` on every "not idle" pass
- Fix: Timer only updated on real work abort, `auto_default_started_at` preserves sequence start

**Bug 3: `default_ready` is not self-validating** — FIXED
- Was: Returned immediately without validation
- Fix: Lines 1665-1688 re-validate and repair on drift via `AUTO_DEFAULT_DRIFT`

**Bug 4: Waiting states don't block normal resource logic** — FIXED
- Was: Only checked `triggered`
- Fix: Line 1927 now checks `managed` OR `triggered`

**Bug 5: No heartbeat freshness check** — FIXED
- Was: Used raw heartbeat data
- Fix: `_is_fresh_gpu_state()` helper at lines 1402-1415, used for `default_in_target`

**Fully implemented.** Freshness checks now applied consistently:
- `_is_fresh_gpu_state()` uses `self.heartbeat_stale_seconds` by default (line 1402-1420)
- `_is_system_globally_idle()` skips stale heartbeats for active_tasks/meta_task (line 1520-1527)
- `_has_real_work_in_flight()` skips stale heartbeats (line 1463-1475)
- `loaded_split_groups` only includes fresh heartbeats (line 1628-1637)
- `single_hot_needing_unload` only includes fresh heartbeats (line 1643-1660)
- Stale heartbeats cannot block idle detection, trigger aborts, or create unload decisions

---

## Remaining Fixes — ALL COMPLETE

All remaining fixes have been implemented:

### 1. Part A: Token verification fail closed — DONE
- `/home/bryan/llm_orchestration/shared/agents/gpu_split.py:1243-1257`
- Reservation with token requires owner to have matching token
- Missing or mismatched tokens cause verification failure

### 2. Part B: `_stop_split_runtime()` uses owner metadata first — DONE
- `/home/bryan/llm_orchestration/shared/agents/gpu_split.py:1767-1822`
- Reads owner meta to get group_id
- Calls `_force_kill_split_runtime_owner()` before Popen fallback
- Survives parent-process death

### 3. Part C: Freshness checks applied consistently — DONE
- `_is_fresh_gpu_state()` uses `self.heartbeat_stale_seconds` by default
- Idle detection skips stale heartbeats
- Real-work detection skips stale heartbeats
- Split group and single hot classification skips stale heartbeats

### 4. Document cleanup — DONE
- Status table updated to COMPLETE
- Contradictory notes removed
- Single source of truth maintained

---

## Exit Criteria — ALL MET

1. ✅ Owner-meta token verification fails closed on any missing/mismatched token
2. ✅ `_stop_split_runtime()` kills by owner metadata first
3. ✅ Auto-default uses fresh heartbeat gating consistently for:
   - ✅ idle detection
   - ✅ real-work detection
   - ✅ split/single normalization classification
   - ✅ default target verification
4. ✅ Document cleaned of contradictory status claims

---

#### Actionable Fix Checklist — ALL DONE

**File: `/home/bryan/llm_orchestration/shared/agents/brain.py`**

- [x] Add `self.auto_default_started_at: Optional[datetime] = None` — line 209

**File: `/home/bryan/llm_orchestration/shared/agents/brain_resources.py`**

- [x] **Fix 1**: Add `result["managed"] = False` to initial result dict — line 1544
- [x] **Fix 2**: Add helper `_is_fresh_gpu_state()` — lines 1402-1415
- [x] **Fix 3**: Add helper `_has_real_work_in_flight()` — lines 1417-1468
- [x] **Fix 4**: Rewrite "not is_idle" block to distinguish real work — lines 1563-1600
  - If real work: reset phase, timer, and `started_at`
  - If only auto-default meta tasks: set `managed=True`, don't reset
- [x] **Fix 5**: Self-validate `default_ready` — lines 1665-1688
  - Recompute `system_normalized` and `default_in_target`
  - If drift detected: log `AUTO_DEFAULT_DRIFT`, demote to normalizing
- [x] **Fix 6**: Use `_is_fresh_gpu_state()` for `default_in_target` — lines 1636-1658
- [x] **Fix 7**: Short-circuit on `managed` or `triggered` — line 1927
- [x] **Fix 8**: Set `auto_default_started_at` when entering normalizing — lines 1766, 1809
- [x] **Fix 9**: Clear `auto_default_started_at` on abort/complete — lines 1579, 1697, 1752
- [x] **Fix 10**: Add log events — implemented:
  - `AUTO_DEFAULT_MANAGED_WAIT` (line 1590)
  - `AUTO_DEFAULT_REAL_WORK_ABORT` (line 1573)
  - `AUTO_DEFAULT_DRIFT` (line 1676)
  - `AUTO_DEFAULT_STALE_HEARTBEAT` (line 1654)

---

#### Expected Final Behavior

1. System becomes fully idle
2. Idle timer elapses once (90s default)
3. Phase A issues unloads → enters `"normalizing"`
4. While unloads in flight, auto-default keeps ownership (`managed=True`)
5. Once normalized, Phase B issues `load_llm` for gpu-2
6. While load in flight, auto-default still owns the rig
7. Once gpu-2 has fresh heartbeat with `qwen2.5:7b`, phase → `"default_ready"`
8. If drift occurs while idle, auto-default self-heals
9. If real work appears, auto-default aborts cleanly and resets

---

#### Legacy Documentation (Reference Only)

The sections below are kept for historical context but superseded by the checklist above.
  - no active GPU tasks / GPU meta tasks in heartbeats

Implemented behavior that is working:
- Idle timer resets when system becomes busy
- Auto-default is one-shot per idle period via `auto_default_active`
- Auto-default issues targeted `unload_llm` and targeted `load_llm`
- Auto-default can target `gpu-2` with `qwen2.5:7b`

Still needs correction (important):

1. Split unloads are iterated per GPU, not per split group
- Current code loops over `split_loaded`, which is a list of GPUs with split models loaded.
- This means the same split group can be processed twice (once for each member).
- `_insert_resource_task()` dedup may suppress duplicates, but the policy should be correct by construction.
- Required fix:
  - Build a unique set of loaded split `group_id`s first.
  - Issue at most one targeted `unload_split_llm` per group.

2. Normalize and default-load happen in the same trigger pass
- Current code can enqueue:
  - `unload_split_llm`
  - `unload_llm`
  - `load_llm` for default
  all in one pass.
- This is not authoritative sequencing.
- The default `load_llm` should only be queued after normalization is complete.
- Required fix:
  - Make auto-default a true phased state machine:
    - Phase A: normalize to cold
    - Phase B: only after no split groups remain and no non-default hot workers remain, enqueue default `load_llm`
  - Do not queue Phase B work in the same cycle as Phase A unloads.

3. `auto_default_active` is set too early
- Current code sets `self.auto_default_active = True` immediately after issuing tasks.
- If normalization is still in progress, this suppresses follow-up auto-default logic in later idle cycles.
- Required fix:
  - Set `auto_default_active = True` only after the system is actually in default state, or
  - Replace the boolean with a small explicit auto-default phase tracker:
    - `None`
    - `normalizing`
    - `default_ready`

4. Action reporting is lossy
- `result["action"]` is overwritten as multiple tasks are queued.
- This makes logs harder to trust.
- Required fix:
  - Replace `action` with ordered `actions: []`
  - Record each queued resource action explicitly

5. Current implementation can race with resource cooldown timing
- Because unload and load tasks are inserted in the same pass, cooldown/dedup behavior may create non-deterministic ordering pressure.
- Required fix:
  - Phase separation (above) is the authoritative solution; do not rely on cooldown timing to serialize behavior.

Recommended final policy:
- `idle_detected` -> queue normalize tasks only
- while normalize tasks pending or split/single hot mismatches remain: do nothing else
- once normalized: queue one targeted default `load_llm`
- once `gpu-2` is `ready_single` with `qwen2.5:7b`: mark auto-default complete

#### Verification Update (Current Live Code)

Re-checked against:
- `/home/bryan/llm_orchestration/shared/agents/brain.py`
- `/home/bryan/llm_orchestration/shared/agents/brain_resources.py`

What is correctly implemented now:
- `brain.py` uses `auto_default_phase: Optional[str] = None`
- `_check_auto_default_policy()` returns `actions: []` instead of a lossy single `action`
- split unload issuance is deduplicated by `runtime_group_id`
- unload-only normalization and default `load_llm` are separated inside the function body
- `default_ready` is only set when the default GPU is actually in the target loaded state

What still needs changing:

1. Phase state is reset while normalization/load tasks are running
- `_is_system_globally_idle()` treats queued/processing meta tasks as "not idle".
- `_check_auto_default_policy()` resets both:
  - `self.auto_default_last_busy_at = now`
  - `self.auto_default_phase = None`
  whenever the system is not idle.
- Result:
  - after Phase A queues unloads, the next brain cycle sees pending meta work, marks the system busy, and clears `"normalizing"`
  - after Phase B queues `load_llm`, the next brain cycle also clears phase state
- This means the phase machine does not actually persist across the normalize/load sequence; it re-enters from scratch after each meta-task burst and waits for a fresh idle window again.
- Required fix:
  - when `auto_default_phase in {"normalizing", "default_ready"}`, do not blindly clear phase on "not idle"
  - distinguish:
    - expected busy caused by auto-default-owned meta tasks
    - real busy caused by new workload / user activity
  - only reset to `None` when real work resumes

2. Idle timer is restarted by auto-default's own work
- Because `auto_default_last_busy_at` is updated on every "not idle" pass, auto-default's own unload/load tasks restart the idle timer.
- Result:
  - even after normalization completes, Phase B may be delayed by another full `auto_default_idle_seconds`
  - after `load_llm` completes, final convergence to `default_ready` can also be delayed
- Required fix:
  - do not update `auto_default_last_busy_at` while the only busy reason is the in-flight auto-default sequence
  - preserve the original idle trigger timestamp until the state machine completes or real work arrives

3. `default_ready` is sticky and not self-healing
- If `self.auto_default_phase == "default_ready"`, the function returns immediately.
- It does not re-check whether:
  - `gpu-2` still has `qwen2.5:7b`
  - the system is still normalized
- Result:
  - if the default GPU drifts while the system remains otherwise idle, the policy will not repair it until some other busy period resets the phase.
- Required fix:
  - in `"default_ready"`, verify `system_normalized and default_in_target`
  - if either is false, demote phase back to `None` or `"normalizing"` and repair

4. Waiting states do not block normal resource logic
- `_make_resource_decisions()` only short-circuits when `auto_default_result["triggered"]` is true.
- `_check_auto_default_policy()` returns `triggered=False` in passive wait states:
  - while idle timer is counting down
  - while `"normalizing"` is waiting for unload completion
  - while `"default_ready"` is already set
- Result:
  - the generic resource logic below auto-default still runs in those cycles
  - this weakens the intended "auto-default owns the rig until complete" policy
  - today this is partially masked because the rig is usually empty at that point, but it is not authoritative behavior
- Required fix:
  - return a separate state such as `active=True` or `handled=True` whenever auto-default is in a managed phase (`"normalizing"` or `"default_ready"`)
  - `_make_resource_decisions()` should short-circuit on managed-phase ownership, not only on "queued an action this cycle"

5. No heartbeat freshness check in default-state evaluation
- `_check_auto_default_policy()` uses raw `gpu_states` from `_get_gpu_states()`, which reads heartbeat files directly.
- `_get_gpu_states()` does not filter stale heartbeats.
- Result:
  - stale `loaded_model`, `runtime_placement`, `active_tasks`, or `meta_task_active` fields can block or misdirect auto-default
  - a stale heartbeat can falsely:
    - prevent idle detection forever
    - mark the system normalized when it is not
    - mark the default GPU "in target" when it is not
- Required fix:
  - evaluate heartbeat age before trusting each GPU state
  - for auto-default decisions, either:
    - ignore stale heartbeats entirely, or
    - treat stale heartbeats as non-ready / unknown and fail closed
  - specifically, `default_in_target` should require a fresh heartbeat, not just matching cached fields

Conclusion:
- The function has the right structure now, but the summary "all 5 issues fixed" is too strong.
- The queuing behavior is improved, but the state machine still needs another pass so auto-default can survive its own meta-task activity, own the rig while active, and converge deterministically from fresh state only.

#### Authoritative Fix Path (Remaining Work)

This is the clean implementation path for the remaining auto-default issues. Do this as one coherent pass in `brain_resources.py`, not as isolated patches.

1. Introduce explicit auto-default ownership in `_check_auto_default_policy()`
- Keep the existing `result["triggered"]` for "queued work this cycle".
- Add a second field:
  - `result["managed"] = bool`
- Meaning:
  - `managed=True` whenever auto-default is actively responsible for system convergence
  - this includes:
    - phase `"normalizing"`
    - phase `"default_ready"`
    - optionally the final idle countdown window if you want auto-default to fully own that stage too
- Implementation:
  - initialize:
    ```python
    result = {"triggered": False, "managed": False, "actions": [], "details": {}}
    ```
  - when `current_phase in {"normalizing", "default_ready"}`, set `result["managed"] = True` before returning

2. Stop resetting phase/timer for auto-default-owned busy states
- Current reset block in `/home/bryan/llm_orchestration/shared/agents/brain_resources.py:1493` is too broad.
- Replace it with a two-branch decision:
  - Branch A: busy because real work resumed
    - active batch
    - queue/processing has non-meta work
    - private tasks exist
    - GPU active tasks indicate real execution
    - then reset:
      - `self.auto_default_last_busy_at = now`
      - `self.auto_default_phase = None`
  - Branch B: busy only because auto-default meta tasks are in flight
    - pending `unload_llm`, `unload_split_llm`, or targeted default `load_llm`
    - do not reset phase
    - do not update `auto_default_last_busy_at`
    - return `managed=True`
- Cleanest way:
  - add helper:
    - `_classify_auto_default_busy_reasons(queue_stats, demand_window, gpu_states, not_idle_reasons) -> {"real_work": bool, "auto_default_only": bool, "reasons": [...]}``
  - use it inside `_check_auto_default_policy()`

3. Preserve the original idle trigger timestamp until convergence completes
- Once auto-default first triggers, the idle timer should be considered "latched".
- Add state field in `brain.py`:
  - `self.auto_default_started_at: Optional[datetime] = None`
- Behavior:
  - when Phase A first starts:
    - set `self.auto_default_started_at = now`
  - while phase is `"normalizing"`:
    - do not overwrite `auto_default_last_busy_at`
  - when real work resumes:
    - clear both:
      - `self.auto_default_phase = None`
      - `self.auto_default_started_at = None`
  - when convergence reaches `"default_ready"`:
    - keep phase at `"default_ready"`
    - `auto_default_started_at` may remain for observability or be cleared explicitly
- This prevents the system from requiring a second full idle window after its own unload/load tasks.

4. Make `"default_ready"` self-validating instead of terminal-blind
- Current code at `/home/bryan/llm_orchestration/shared/agents/brain_resources.py:1555` returns immediately.
- Replace with:
  - recompute:
    - `system_normalized`
    - `default_in_target`
  - if both true:
    - return with `managed=True`
  - if either false:
    - log `AUTO_DEFAULT_DRIFT`
    - demote phase:
      - to `"normalizing"` if unloads are needed
      - to `None` if you want to re-enter through the normal trigger path
    - continue repair path instead of returning
- Recommended:
  - if normalized but default GPU drifted: queue targeted `load_llm` and move to `"normalizing"`
  - if not normalized: queue normalization unloads and stay `"normalizing"`

5. Treat stale heartbeats as untrusted input
- `_get_gpu_states()` in `/home/bryan/llm_orchestration/shared/agents/brain_monitor.py:665` returns raw heartbeat JSON.
- Do not change the global helper unless desired; fix it locally in auto-default logic first.
- In `_check_auto_default_policy()`:
  - compute heartbeat age from `last_heartbeat` / `updated_at` / timestamp field already present in the heartbeat payload
  - compare to `self.heartbeat_stale_seconds`
- Rule:
  - stale heartbeat cannot establish:
    - `default_in_target`
    - `system_normalized`
  - stale heartbeat should be treated as unknown / not-ready
- Practical implementation:
  - add helper in `brain_resources.py`:
    - `_is_fresh_gpu_state(state: Dict[str, Any]) -> bool`
  - use only fresh states when building:
    - `loaded_split_groups`
    - `single_hot_needing_unload`
    - `default_in_target`
  - if the default GPU heartbeat is stale, fail closed:
    - `default_in_target = False`

6. Short-circuit `_make_resource_decisions()` on auto-default management, not only task insertion
- Current code at `/home/bryan/llm_orchestration/shared/agents/brain_resources.py:1777` only returns early if `triggered=True`.
- Change to:
  ```python
  if auto_default_result.get("managed") or auto_default_result.get("triggered"):
      return
  ```
- This makes auto-default authoritative while it is active, including passive waiting phases.

7. Separate "real work" from auto-default meta tasks in idle detection
- `_is_system_globally_idle()` is still useful, but it is too coarse to drive resets by itself.
- Keep it as the first coarse filter.
- Add a second helper specifically for reset decisions:
  - `_has_real_work_in_flight(queue_stats, demand_window, gpu_states) -> bool`
- It should ignore:
  - auto-default-owned meta tasks
- It should count:
  - non-meta queued tasks
  - processing non-meta tasks
  - private tasks
  - active batch entries
  - GPU `active_tasks`
- Use this helper to decide when to clear phase state.

8. Tighten observability so behavior is debuggable
- Add explicit log events:
  - `AUTO_DEFAULT_MANAGED_WAIT`
  - `AUTO_DEFAULT_REAL_WORK_ABORT`
  - `AUTO_DEFAULT_DRIFT`
  - `AUTO_DEFAULT_STALE_HEARTBEAT`
- Include:
  - `phase`
  - `not_idle_reasons`
  - `real_work`
  - `auto_default_only`
  - `idle_seconds`
  - `started_at`

#### Minimal Edit Checklist

Files:
- `/home/bryan/llm_orchestration/shared/agents/brain.py`
- `/home/bryan/llm_orchestration/shared/agents/brain_resources.py`

In `brain.py`:
- add `self.auto_default_started_at: Optional[datetime] = None`

In `brain_resources.py`:
- extend `_check_auto_default_policy()` result with `managed`
- add `_is_fresh_gpu_state(...)`
- add `_has_real_work_in_flight(...)`
- optionally add `_classify_auto_default_busy_reasons(...)`
- rewrite the `if not is_idle:` block so it does not clear state for auto-default-owned meta activity
- rewrite the `current_phase == "default_ready"` block to revalidate and self-heal
- update `_make_resource_decisions()` to early-return when auto-default is managing

#### Expected Final Behavior

1. System becomes fully idle
2. Idle timer elapses once
3. Phase A issues unloads and enters `"normalizing"`
4. While unload tasks are in queue/processing, auto-default keeps ownership and does not reset itself
5. Once normalized, one targeted `load_llm` is issued for `gpu-2`
6. While that load is in flight, auto-default still owns the rig
7. Once `gpu-2` has a fresh heartbeat showing `single_gpu + qwen2.5:7b`, phase becomes `"default_ready"`
8. If the default state drifts while the system stays idle, auto-default repairs it automatically
9. If real work appears at any point, auto-default aborts cleanly and resets its phase

#### 1. Add Config + State Fields (`brain.py` in `__init__`)

```python
self.auto_default_enabled = bool(config.get("auto_default_enabled", True))
self.auto_default_idle_seconds = int(config.get("auto_default_idle_seconds", 90))
self.auto_default_last_busy_at = datetime.now()
self.auto_default_active = False

# Default target
self.auto_default_gpu = "gpu-2"
self.auto_default_model = "qwen2.5:7b"
```

#### 2. Add Strict "System Idle" Gate (`brain_resources.py` in `_make_resource_decisions`)

Compute `is_globally_idle` only if ALL true:
1. `len(self.active_batches) == 0`
2. Queue empty: `queue_stats["total_pending"] == 0`
3. Processing empty: `queue_stats["processing_count"] == 0`
4. No private tasks: `demand_window["total_llm"] == 0` and no private cpu/script/meta tasks
5. No meta in queue/processing: `_has_pending_meta_command(...)` false for load/unload/split
6. No active GPU tasks from heartbeats: `active_tasks` empty, `meta_task_active` false

If any false:
```python
self.auto_default_last_busy_at = now
self.auto_default_active = False
return  # Continue normal logic
```

If true but idle duration < `auto_default_idle_seconds`:
```python
return  # Not idle long enough yet
```

#### 3. Normalize to Cold First, Then Set Default

Once idle long enough, run exactly once per idle period (`auto_default_active` guard):

**Phase A: Normalize**
- Unload all split groups (targeted `unload_split_llm` per loaded group)
- Unload all single hot workers except default gpu (targeted `unload_llm`)

**Phase B: Enforce Default**
```python
if default_gpu is not ready_single with default_model:
    self._insert_resource_task(
        "load_llm",
        meta={
            "target_model": self.auto_default_model,
            "candidate_workers": [self.auto_default_gpu],
        }
    )
```

Set `self.auto_default_active = True` when first triggered.
Reset to `False` when system becomes busy again.

#### 4. Keep It Non-Flappy

- Use existing resource cooldown + dedup (`_insert_resource_task` already does this)
- Never trigger while any batch active
- Never trigger while meta tasks are running

#### 5. Make Targeting Explicit

For all auto-default meta tasks, include `candidate_workers` (and `group_id` for split unloads).
Do not use generic unload/load during auto-default.

#### State Machine

```
Idle -> Normalize -> DefaultReady
  ^                      |
  |______(busy again)____|
```

---

## Files to Modify

| File | Changes |
|------|---------|
| `gpu_constants.py` | Add states, thresholds, reservation status sets |
| `gpu_split.py` | Stability gate, ready_token, fix runtime_already_loaded (L2093-2112), fix follower mirroring (L2216-2258), process group launch (L1253), kill helpers, owner meta enhancement |
| `gpu_tasks.py` | `_validate_split_ready_token()`, `_handle_attestation_miss()`, phased recovery |
| `brain_resources.py` | Check ready_token in `_reconcile_split_group_state()`, auto-default idle gate + normalize logic |
| `brain.py` | Add auto_default config fields + state tracking in `__init__` |

---

## Verification

Test loop on pair 4/5:
1. Cold both GPUs
2. Load split 14B
3. Claim 14B task
4. Repeat 20 cycles

Pass conditions:
- Zero `runtime_already_loaded -> ready` before launcher-ready
- Zero `target_model=None` joins
- Zero `expected_model_missing ... actual=[]` at first claim after ready
- Zero owner-meta mismatch clears during normal operation
- Process group cleanup verified (no orphan llama runners)
