#!/usr/bin/env bash
# GPU rig host readiness: PASS/FAIL per check, exit 1 on any FAIL. Read-only; safe while models run.
# usage: check_rig.sh [--deep]     --deep also starts a CUDA container on every GPU (a few seconds)
# The standard it checks is shared/workspace/systems_prep.md; prepare_rig.sh applies the parts it can.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TREE="$(cd "$HERE/../../.." && pwd)"             # ~/llm_orchestration
BENCH="$TREE/shared/plans/shoulders/benchmarking"
DEEP=0; [ "${1:-}" = "--deep" ] && DEEP=1
DRIVER_BRANCH=580                                  # last branch for the Pascal 1060s; see systems_prep.md
EXPECT_GPUS="${RIG_EXPECT_GPUS:-6}"
pass=0; fail=0
ok()  { echo "PASS  $1"; pass=$((pass+1)); }
bad() { echo "FAIL  $1${2:+: $2}"; fail=$((fail+1)); }
chk() { local name=$1; shift; if "$@" >/dev/null 2>&1; then ok "$name"; else bad "$name"; fi; }
same() { [ -f "$1" ] && [ -f "$2" ] && cmp -s "$1" "$2"; }

# --- driver ---
pkg=$(dpkg-query -W -f='${Version}' "nvidia-driver-$DRIVER_BRANCH" 2>/dev/null)
[ -n "$pkg" ] && ok "nvidia-driver-$DRIVER_BRANCH installed ($pkg)" || bad "nvidia-driver-$DRIVER_BRANCH installed"
openpkg=$(dpkg-query -W -f='${Package} ' 'nvidia-driver-*-open' 'nvidia-dkms-*-open' 2>/dev/null | xargs)
[ -z "$openpkg" ] && ok "no open kernel module driver (Pascal needs the proprietary one)" || bad "open driver installed" "$openpkg"
mod=$(cat /sys/module/nvidia/version 2>/dev/null)
utils=$(dpkg-query -W -f='${Version}' "nvidia-utils-$DRIVER_BRANCH" 2>/dev/null | cut -d- -f1)
[ -n "$mod" ] && [ "$mod" = "$utils" ] && ok "loaded module $mod matches userspace" \
  || bad "loaded module matches userspace" "module ${mod:-none}, utils ${utils:-none} (reboot after a driver update)"
[ "${mod%%.*}" = "$DRIVER_BRANCH" ] && ok "driver branch $DRIVER_BRANCH" || bad "driver branch $DRIVER_BRANCH" "loaded ${mod:-none}"
lic=$(modinfo -F license nvidia 2>/dev/null)
[ "$lic" = "NVIDIA" ] && ok "proprietary kernel module" || bad "proprietary kernel module" "license ${lic:-unknown}"
dkms_ok=$(dkms status "nvidia/$mod" -k "$(uname -r)" 2>/dev/null | grep -c installed)
[ "${dkms_ok:-0}" -ge 1 ] && ok "DKMS module built for kernel $(uname -r)" || bad "DKMS module built for kernel $(uname -r)"
unheld=$(comm -23 <(dpkg-query -W -f='${db:Status-Abbrev} ${Package}\n' '*nvidia*' 2>/dev/null |
  awk -v b="$DRIVER_BRANCH" '$1 ~ /^.i/ && $2 ~ b {sub(/:.*/, "", $2); print $2}' | sort -u) <(apt-mark showhold | sed 's/:.*//' | sort -u) | xargs)
[ -z "$unheld" ] && ok "driver $DRIVER_BRANCH packages held" || bad "driver $DRIVER_BRANCH packages held" "not held: $unheld (prepare_rig.sh --apply)"

# --- GPUs ---
n=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ')
[ "$n" = "$EXPECT_GPUS" ] && ok "nvidia-smi sees $n GPUs" || bad "nvidia-smi sees $EXPECT_GPUS GPUs" "saw $n"
pm=$(nvidia-smi --query-gpu=persistence_mode --format=csv,noheader 2>/dev/null | grep -vc Enabled)
[ "$pm" = "0" ] && ok "persistence mode on every GPU" || bad "persistence mode on every GPU" "$pm off"
offplan=$(python3 "$HERE/hw_power.py" show --json 2>/dev/null | python3 -c '
import json, sys
rows = json.load(sys.stdin)
print(" ".join("GPU%s %s/%s W" % (r["index"], r["limit_w"], r["planned_w"]) for r in rows if r["limit_w"] != r["planned_w"]))' 2>/dev/null)
[ $? -eq 0 ] && [ -z "$offplan" ] && ok "power limits match the plan" || bad "power limits match the plan" "${offplan:-hw_power.py failed}"
replays=$(nvidia-smi -q 2>/dev/null | awk '/Replays Since Reset/ {s += $NF} END {print s+0}')
echo "INFO  PCIe link replays since reset (all GPUs): $replays (compare before/after a split load)"

# --- updates ---
for t in apt-daily.timer apt-daily-upgrade.timer; do
  st=$(systemctl is-enabled "$t" 2>/dev/null)
  [ "$st" = "disabled" ] || [ "$st" = "masked" ] && ok "$t off" || bad "$t off" "$st (unattended-upgrade upgraded the driver on 2026-06-08)"
done

# --- containers ---
chk "docker active" systemctl is-active --quiet docker
chk "bryan in docker group" bash -c 'id -nG bryan | tr " " "\n" | grep -qx docker'
chk "nvidia container toolkit" nvidia-ctk --version
chk "docker nvidia runtime configured" grep -q nvidia-container-runtime /etc/docker/daemon.json
img=$(python3 -c "import json; print(json.load(open('$BENCH/model_registry.json'))['runtime_image'])" 2>/dev/null)
[ -n "$img" ] && docker image inspect "$img" >/dev/null 2>&1 && ok "runtime image $img" || bad "runtime image ${img:-?}" "build it (llama_runtime/README.md)"
missing=""
for d in "$BENCH"/docker/bench-*/; do
  b=$(basename "$d"); docker image inspect "$b:latest" >/dev/null 2>&1 || missing="$missing $b"
done
[ -z "$missing" ] && ok "bench images built" || bad "bench images built" "missing:$missing"
if [ "$DEEP" = 1 ]; then
  cuda=$(docker run --rm --gpus all nvidia/cuda:12.9.2-runtime-ubuntu24.04 nvidia-smi -L 2>/dev/null | grep -c '^GPU ')
  [ "$cuda" = "$EXPECT_GPUS" ] && ok "CUDA container sees $cuda GPUs" || bad "CUDA container sees $EXPECT_GPUS GPUs" "saw $cuda"
fi

# --- storage and memory ---
chk "fstab has the Pi share automount" grep -qE '^10\.0\.0\.2:/media/bryan/shared[[:space:]]+/mnt/shared[[:space:]]+nfs' /etc/fstab
chk "shared drive reachable (/mnt/shared/models)" timeout 20 ls /mnt/shared/models
chk "NVMe hot-set /data/model-hotset" test -d /data/model-hotset
chk "SATA hot-set ~/model-hotset" test -d /home/bryan/model-hotset
extra=$(swapon --noheadings --show=NAME 2>/dev/null | grep -v -E '^(/dev/zram0|/swap.img)$' | xargs)
[ -z "$extra" ] && ok "no extra swap (zram + /swap.img only)" || bad "no extra swap" "$extra (never add swap: RIG_PROJECT run notes section 11)"
chk "zram swap active" bash -c 'swapon --noheadings --show=NAME | grep -qx /dev/zram0'
chk "earlyoom active" systemctl is-active --quiet earlyoom
chk "earlyoom settings match repo" same /etc/default/earlyoom "$HERE/systemd/earlyoom.default"
[ "$(sysctl -n vm.swappiness)" = "10" ] && ok "vm.swappiness 10" || bad "vm.swappiness 10" "$(sysctl -n vm.swappiness)"

# --- rig layer ---
for u in rig-boot rig-guard rig-plan rig-power rig-performance-tune; do
  chk "$u.service matches repo" same "/etc/systemd/system/$u.service" "$HERE/systemd/$u.service"
  chk "$u.service enabled" systemctl is-enabled --quiet "$u"
done
for u in rig-guard rig-plan; do chk "$u running" systemctl is-active --quiet "$u"; done
for u in llm-bench llm-orchestration; do
  chk "$u.service matches repo" same "/etc/systemd/system/$u.service" "$HERE/systemd/$u.service"
done
chk "rig-performance-tune.sh matches repo" same /usr/local/sbin/rig-performance-tune.sh "$HERE/rig-performance-tune.sh"
[ "$(readlink -f /usr/local/bin/rig)" = "$HERE/rig.py" ] && ok "rig command links to rig.py" || bad "rig command links to rig.py"
gov=$(cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null | sort -u | xargs)
[ "$gov" = "performance" ] && ok "CPU governor performance" || bad "CPU governor performance" "$gov"
lvl=$(python3 -c "import json; print(json.load(open('$TREE/shared/brain/hw/guard_state.json'))['level'])" 2>/dev/null)
[ "$lvl" = "ok" ] && ok "guard ok" || bad "guard ok" "${lvl:-no guard state}"
chk "orchestration venv (requests, filelock)" /home/bryan/llm-orchestration-venv/bin/python -c 'import requests, filelock'
chk "machine config shared/agents/config.json" test -s "$TREE/shared/agents/config.json"
chk "llm-gateway deployed (remote_profiles.json)" test -s /home/bryan/llm-gateway/remote_profiles.json

# --- access and time ---
chk "passwordless sudo" sudo -n true
chk "ssh to the Pi (key, no prompt)" ssh -o BatchMode=yes -o ConnectTimeout=5 pi true
chk "chrony source is the Pi" grep -qx 'server 10.0.0.2 iburst' /etc/chrony/sources.d/pi.sources
src=$(chronyc -n sources 2>/dev/null | awk '/^\^\*/ {print $2}')
[ "$src" = "10.0.0.2" ] && ok "clock synced to the Pi" || bad "clock synced to the Pi" "selected source ${src:-none}"

echo "$pass passed, $fail failed"
[ "$fail" = 0 ]
