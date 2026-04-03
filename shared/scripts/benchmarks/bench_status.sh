#!/bin/bash
# Quick status check for all benchmark activity on the rig
# Usage: ssh 10.0.0.3 'bash /mnt/shared/scripts/benchmarks/bench_status.sh [--deep|--debug|--results]'
#   or locally: bash /media/bryan/shared/scripts/benchmarks/bench_status.sh [--deep|--debug|--results]

set -euo pipefail

DEEP=0
RESULTS=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --deep|--debug)
      DEEP=1
      ;;
    --results)
      RESULTS=1
      ;;
    -h|--help)
      echo "Usage: bash bench_status.sh [--deep|--debug|--results]"
      echo "  default: fast rig overview"
      echo "  --deep : include per-port runtime probes and benchmark progress diagnostics"
      echo "  --results : summarize recent benchmark result files and task completion state"
      exit 0
      ;;
    *)
      echo "Unknown arg: $1"
      exit 1
      ;;
  esac
  shift
done

# Colors
G='\033[0;32m'; R='\033[0;31m'; Y='\033[0;33m'; C='\033[0;36m'; B='\033[1m'; D='\033[0;90m'; N='\033[0m'

# Detect path prefix (rig vs laptop)
if [ -d /mnt/shared/gpus ]; then
  SHARED=/mnt/shared
elif [ -d /media/bryan/shared/gpus ]; then
  SHARED=/media/bryan/shared
else
  echo "Cannot find shared drive (tried /mnt/shared and /media/bryan/shared)"
  exit 1
fi

# Helper: time ago from ISO timestamp
time_ago() {
  local ts="$1"
  # Handle ISO timestamps with or without timezone
  local clean=$(echo "$ts" | sed 's/\.[0-9]*//')
  local then_epoch=$(date -d "$clean" +%s 2>/dev/null) || return
  local now_epoch=$(date +%s)
  local diff=$((now_epoch - then_epoch))
  if [ "$diff" -lt 60 ]; then
    echo "${diff}s ago"
  elif [ "$diff" -lt 3600 ]; then
    echo "$((diff / 60))m ago"
  elif [ "$diff" -lt 86400 ]; then
    echo "$((diff / 3600))h $((diff % 3600 / 60))m ago"
  else
    echo "$((diff / 86400))d ago"
  fi
}

# Helper: extract JSON value (simple jq-free parser for flat keys)
jval() {
  local file="$1" key="$2"
  python3 -c "import json,sys; d=json.load(open('$file')); print(d.get('$key',''))" 2>/dev/null
}

jval_nested() {
  local file="$1" expr="$2"
  python3 -c "import json,sys; d=json.load(open('$file')); print($expr)" 2>/dev/null
}

# Helper: estimate benchmark container progress and ETA
bench_progress() {
  local cname="$1"
  local logs started_at now_epoch start_epoch elapsed

  # Get container start time
  started_at=$(docker inspect "$cname" --format '{{.State.StartedAt}}' 2>/dev/null) || return
  start_epoch=$(date -d "$started_at" +%s 2>/dev/null) || return
  now_epoch=$(date +%s)
  elapsed=$((now_epoch - start_epoch))
  [ "$elapsed" -lt 10 ] && return  # too early to estimate

  logs=$(docker logs "$cname" 2>&1)

  local done_count=0 total=0 suite=""

  if [[ "$cname" == bench-code-* ]]; then
    # Code benchmark: HumanEval (164) + Mbpp (378 base) = 542 codegen calls
    done_count=$(echo "$logs" | grep -c "Codegen:" || true)
    total=542
    suite="code"
  elif [[ "$cname" == bench-pipeline-* ]]; then
    # Pipeline: count completed test lines
    done_count=$(echo "$logs" | grep -cE "^(PASS|FAIL|ERROR|json_schema|command_safety|ambiguity|tool_plan|orchestration|long_context)" || true)
    # Pipeline has 6 subtests, ~60-80 individual checks, but results vary
    # Use test-level counting from status.json if available
    local test_count=$(echo "$logs" | grep -cE "Running test:" || true)
    local test_done=$(echo "$logs" | grep -cE "(PASS|FAIL)" || true)
    if [ "$test_count" -gt 0 ]; then
      total=$test_count
      done_count=$test_done
      suite="pipeline"
    else
      return  # can't estimate pipeline reliably without counts
    fi
  elif [[ "$cname" == bench-reasoning-* ]]; then
    # Reasoning: gsm8k + bbh + drop, each with variable item counts
    # Count request completions
    local gsm_done=$(echo "$logs" | grep -c "gsm8k" || true)
    local bbh_done=$(echo "$logs" | grep -c "bbh" || true)
    local drop_done=$(echo "$logs" | grep -c "drop" || true)
    # Check which sub-benchmarks are running via log patterns
    done_count=$(echo "$logs" | grep -cE "Requesting|Running|completed" || true)
    # Reasoning totals depend on limit, check for limit in logs
    local limit=$(echo "$logs" | grep -oP 'limit[= ]+\K[0-9]+' | head -1)
    limit=${limit:-5}
    # gsm8k=limit, bbh=limit*num_tasks(~27), drop=limit — rough estimate
    total=$(( limit + limit * 27 + limit ))
    suite="reasoning"
  fi

  [ "$total" -eq 0 ] && return
  [ "$done_count" -eq 0 ] && return

  local pct=$((done_count * 100 / total))
  local rate_per_sec=$(python3 -c "print(f'{$done_count / $elapsed:.2f}')" 2>/dev/null)
  local remaining=$((total - done_count))
  local eta_sec=$(python3 -c "r=$done_count/$elapsed; print(int($remaining/r)) if r>0 else print(0)" 2>/dev/null)

  # Format ETA
  local eta_str
  if [ "$eta_sec" -lt 60 ]; then
    eta_str="${eta_sec}s"
  elif [ "$eta_sec" -lt 3600 ]; then
    eta_str="$((eta_sec / 60))m"
  else
    eta_str="$((eta_sec / 3600))h $((eta_sec % 3600 / 60))m"
  fi

  echo "${suite}: ${done_count}/${total} (${pct}%) ~${eta_str} remaining"
}

bench_progress_line() {
  local cname="$1"
  docker logs "$cname" 2>&1 | grep -E "Codegen:|Running|Requesting|completed" | tail -1 | sed 's/^[[:space:]]*//'
}

bench_error_summary() {
  local cname="$1"
  local logs api_wait runtime_unreach oom
  logs=$(docker logs "$cname" 2>&1 || true)
  api_wait=$(echo "$logs" | grep -c "API connection error\. Waiting" || true)
  runtime_unreach=$(echo "$logs" | grep -c "Cannot reach llama-compatible runtime" || true)
  oom=$(echo "$logs" | grep -cE "OutOfMemory|OOM|exit code 137|Killed" || true)
  [ "$api_wait" -eq 0 ] && [ "$runtime_unreach" -eq 0 ] && [ "$oom" -eq 0 ] && return 0
  echo "api_wait=${api_wait} runtime_unreachable=${runtime_unreach} oom_signals=${oom}"
}

port_probe() {
  local port="$1"
  local raw
  raw=$(curl -s --max-time 2 "http://127.0.0.1:${port}/v1/models" 2>/dev/null || true)
  if [ -z "$raw" ]; then
    echo "down"
    return
  fi
  if echo "$raw" | grep -q '"code":503'; then
    echo "loading"
    return
  fi
  python3 - "$raw" <<'PY'
import json, sys
raw = sys.argv[1]
try:
    data = json.loads(raw)
except Exception:
    print("up (unparsed)")
    raise SystemExit(0)
models = data.get("data") or data.get("models") or []
if models:
    first = models[0]
    mid = first.get("id") or first.get("model") or first.get("name") or "unknown"
    print(f"up {mid}")
else:
    print("up (no model listed)")
PY
}

results_summary() {
  local suite_dir="$1"
  local since_minutes="${2:-720}"
  local limit_runs="${3:-12}"
  python3 - "$suite_dir" "$since_minutes" <<'PY'
import json
import os
import sys
import time
from pathlib import Path

suite_dir = Path(sys.argv[1])
since_minutes = int(sys.argv[2])
limit_runs = int(sys.argv[3]) if len(sys.argv) > 3 else 12
cutoff = time.time() - since_minutes * 60

if not suite_dir.exists():
    raise SystemExit(0)

def fmt_age(ts):
    diff = int(time.time() - ts)
    if diff < 60:
        return f"{diff}s"
    if diff < 3600:
        return f"{diff // 60}m"
    if diff < 86400:
        return f"{diff // 3600}h{(diff % 3600) // 60}m"
    return f"{diff // 86400}d"

status_files = sorted(suite_dir.glob("bench-*/*/status.json"))
seen = {}
for path in status_files:
    try:
        if path.stat().st_mtime < cutoff:
            continue
    except FileNotFoundError:
        continue
    run_dir = path.parent.parent
    task = path.parent.name
    try:
        data = json.load(open(path, "r", encoding="utf-8"))
    except Exception:
        continue
    state = str(data.get("state", "") or "").strip() or "unknown"
    generated = data.get("generated")
    expected = data.get("expected")
    updated_at = path.stat().st_mtime
    eval_files = list(path.parent.glob("*_eval_results.json"))
    score = ""
    if eval_files:
        try:
            eval_data = json.load(open(eval_files[0], "r", encoding="utf-8"))
            p1 = eval_data.get("pass@1", {})
            base = p1.get("base")
            plus = p1.get("plus")
            if base is not None or plus is not None:
                score = f" base={base} plus={plus}"
        except Exception:
            pass
    seen.setdefault(run_dir.name, {"latest": 0.0, "tasks": []})
    seen[run_dir.name]["latest"] = max(seen[run_dir.name]["latest"], updated_at)
    seen[run_dir.name]["tasks"].append(
        (task, state, generated, expected, fmt_age(updated_at), score)
    )

ordered = sorted(
    seen.items(),
    key=lambda item: item[1]["latest"],
    reverse=True,
)[:limit_runs]

for run_name, payload in ordered:
    tasks = payload["tasks"]
    states = [t[1] for t in tasks]
    overall = "completed" if states and all(s == "evaluated" for s in states) else "incomplete"
    print(f"{run_name} [{overall}]")
    for task, state, generated, expected, age, score in sorted(tasks):
        counts = ""
        if generated is not None or expected is not None:
            counts = f" {generated}/{expected}"
        print(f"  {task}: {state}{counts} updated={age}{score}")
PY
}

echo -e "${B}=== GPU Heartbeats ===${N}"
for gpu_dir in "$SHARED"/gpus/gpu_*; do
  hb="$gpu_dir/heartbeat.json"
  [ -f "$hb" ] || continue

  gpu_name=$(jval "$hb" "name")
  state=$(jval "$hb" "state")
  runtime_state=$(jval "$hb" "runtime_state")
  model=$(jval "$hb" "loaded_model")
  placement=$(jval "$hb" "runtime_placement")
  group=$(jval "$hb" "runtime_group_id")
  healthy=$(jval "$hb" "runtime_healthy")
  health_ok=$(jval_nested "$hb" "d.get('runtime_health',{}).get('healthy','')")
  temp=$(jval "$hb" "temperature_c")
  vram_pct=$(jval "$hb" "vram_percent")
  gpu_util=$(jval "$hb" "gpu_util_percent")
  active_workers=$(jval "$hb" "active_workers")
  thermal=$(jval "$hb" "thermal_constrained")
  thermal_pause=$(jval "$hb" "thermal_pause_active")
  owner=$(jval "$hb" "heartbeat_owner")
  updated=$(jval "$hb" "last_updated")

  # Active task details
  task_info=$(jval_nested "$hb" "'; '.join([f\"{t['task_class']}:{t['task_name']} ({t['phase']})\" for t in d.get('active_tasks',[])])")

  # State color
  case "$state" in
    hot) sc=$G ;;
    cold) sc=$D ;;
    *) sc=$Y ;;
  esac

  # Health indicator
  if [ "$health_ok" = "True" ]; then
    hi="${G}OK${N}"
  elif [ "$healthy" = "True" ]; then
    hi="${Y}~OK${N}"
  else
    hi="${R}DOWN${N}"
  fi

  # Thermal indicator
  ti=""
  [ "$thermal" = "True" ] && ti=" ${R}THERMAL${N}"
  [ "$thermal_pause" = "True" ] && ti=" ${R}THERMAL-PAUSED${N}"

  # Placement label
  pl="single"
  [ "$placement" = "split_gpu" ] && pl="split($group)"

  # Staleness
  age=$(time_ago "$updated")

  echo -e "  ${sc}${gpu_name}${N} [${sc}${state}${N}] ${model:-no model} (${pl}) health:${hi}${ti}"
  echo -e "    runtime: ${C}${runtime_state:-idle}${N}  owner: ${owner:-none}  vram: ${vram_pct}%  util: ${gpu_util}%  temp: ${temp}C  updated: ${D}${age}${N}"
  if [ -n "$task_info" ]; then
    echo -e "    tasks: ${Y}${task_info}${N}"
  fi
done

echo ""
echo -e "${B}=== Brain ===${N}"
brain_hb="$SHARED/heartbeats/brain.json"
if [ -f "$brain_hb" ]; then
  brain_state=$(jval "$brain_hb" "state")
  brain_gpus=$(jval_nested "$brain_hb" "d.get('brain_gpus',[])")
  brain_batches=$(jval "$brain_hb" "active_batches")
  brain_updated=$(jval "$brain_hb" "last_updated")
  brain_age=$(time_ago "$brain_updated")
  echo -e "  state: ${brain_state}  gpus: ${brain_gpus}  active_batches: ${brain_batches}  updated: ${D}${brain_age}${N}"
else
  echo "  (no brain heartbeat found)"
fi

echo ""
echo -e "${B}=== Running Containers ===${N}"
running=$(docker ps --format '{{.Names}}|{{.Status}}' 2>/dev/null || true)
if [ -z "$running" ]; then
  echo "  (none)"
else
  echo "$running" | while IFS='|' read -r name status; do
    if [[ "$name" == bench-* ]]; then
      last=$(docker logs "$name" 2>&1 | grep -E "Codegen:|Running|Requesting|gsm8k|bbh|drop|completed|error" | tail -1 | sed 's/^[[:space:]]*//')
      echo -e "  ${Y}${name}${N}  ${status}"
      [ -n "$last" ] && echo -e "    └─ ${last}"

      # Benchmark progress estimation
      bench_eta=$(bench_progress "$name" 2>/dev/null)
      [ -n "$bench_eta" ] && echo -e "    └─ ${C}${bench_eta}${N}"
    elif [[ "$name" == llama-* ]]; then
      echo -e "  ${C}${name}${N}  ${status}"
    else
      echo -e "  ${name}  ${status}"
    fi
  done
fi

echo ""
echo -e "${B}=== Chain Logs ===${N}"
found_logs=0
for f in /tmp/*_chain.log /tmp/*_retry.log; do
  [ -f "$f" ] || continue
  found_logs=1
  base=$(basename "$f")
  last=$(tail -1 "$f" 2>/dev/null)
  echo -e "  ${base}: ${last}"
done
[ "$found_logs" -eq 0 ] && echo "  (none)"

echo ""
echo -e "${B}=== Memory ===${N}"
free -h | awk 'NR==2{printf "  RAM: %s used / %s total (%s avail)\n", $3, $2, $7}'
swapon --show --noheadings 2>/dev/null | while read -r name type size used prio; do
  echo "  Swap: $name ($type) $used / $size"
done

echo ""
echo -e "${B}=== Earlyoom ===${N}"
earlyoom_line=$(journalctl -u earlyoom --no-pager -n 1 2>/dev/null | grep -v "^--" | tail -1 || true)
if [ -n "$earlyoom_line" ]; then
  echo "  $earlyoom_line"
else
  echo "  (unavailable)"
fi

echo ""
echo -e "${B}=== Recent OOM Kills ===${N}"
oom=$(sudo dmesg 2>/dev/null | grep "Out of memory: Killed" | tail -3 || true)
if [ -z "$oom" ]; then
  echo -e "  ${G}(none)${N}"
else
  echo "$oom" | while read -r line; do
    proc=$(echo "$line" | grep -oP 'Killed process \d+ \(\K[^)]+')
    rss=$(echo "$line" | grep -oP 'anon-rss:\K[0-9]+')
    echo -e "  ${R}OOM killed: ${proc} (RSS: ${rss}kB)${N}"
  done
fi

if [ "$DEEP" -eq 1 ]; then
  echo ""
  echo -e "${B}=== Deep Runtime Probes ===${N}"
  for port in 11434 11435 11436 11437 11438 11439; do
    echo "  port ${port}: $(port_probe "$port")"
  done

  echo ""
  echo -e "${B}=== Deep Benchmark Diagnostics ===${N}"
  deep_found=0
  while IFS='|' read -r name status; do
    [ -n "$name" ] || continue
    [[ "$name" == bench-* ]] || continue
    deep_found=1
    echo -e "  ${Y}${name}${N}  ${status}"
    progress=$(bench_progress_line "$name" || true)
    [ -n "$progress" ] && echo "    progress: $progress"
    eta=$(bench_progress "$name" 2>/dev/null || true)
    [ -n "$eta" ] && echo "    estimate: $eta"
    errs=$(bench_error_summary "$name" || true)
    [ -n "$errs" ] && echo "    errors: $errs"
  done < <(docker ps --format '{{.Names}}|{{.Status}}' 2>/dev/null || true)
  [ "$deep_found" -eq 0 ] && echo "  (no benchmark containers)"
fi

if [ "$RESULTS" -eq 1 ]; then
  echo ""
  echo -e "${B}=== Recent Results: bench-code ===${N}"
  results_summary "$SHARED/logs/benchmarks/bench-code/history" 1440 16
  echo ""
  echo -e "${B}=== Recent Results: bench-reasoning ===${N}"
  results_summary "$SHARED/logs/benchmarks/bench-reasoning/history" 1440 16
  echo ""
  echo -e "${B}=== Recent Results: bench-pipeline ===${N}"
  results_summary "$SHARED/logs/benchmarks/bench-pipeline/history" 1440 16
fi
