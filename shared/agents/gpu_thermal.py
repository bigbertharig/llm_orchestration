"""GPU agent thermal and resource management mixin.

Extracted from gpu.py to isolate thermal safety, resource monitoring,
and VRAM budget management.
"""

import os
import signal
import subprocess
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from hardware import scan_cpu_temps

from gpu_constants import (
    DEFAULT_CPU_VRAM_COST,
    VRAM_BUDGET_RATIO,
)


class GPUThermalMixin:
    """Mixin providing thermal safety and resource management methods."""

    def _query_gpu_vram(self) -> int:
        """Query total VRAM for this GPU via nvidia-smi. Fallback for configs without vram_mb."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,memory.total',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            for line in result.stdout.strip().split('\n'):
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 2 and int(parts[0]) == self.gpu_id:
                    return int(float(parts[1]))
        except Exception:
            pass
        return 6144  # Last-resort default

    def _get_gpu_stats(self) -> Dict[str, Any]:
        """Get current GPU resource stats via nvidia-smi."""
        try:
            result = subprocess.run([
                'nvidia-smi',
                '--query-gpu=index,temperature.gpu,power.draw,memory.used,memory.total,'
                'utilization.gpu,clocks.sm,clocks_throttle_reasons.active',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=5)

            for line in result.stdout.strip().split('\n'):
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 8:
                    gpu_index = int(parts[0])
                    if gpu_index == self.gpu_id:
                        vram_used = int(parts[3])
                        vram_total = int(parts[4])
                        vram_percent = int(100 * vram_used / vram_total) if vram_total > 0 else 0
                        return {
                            "temperature_c": int(parts[1]),
                            "power_draw_w": float(parts[2]),
                            "vram_used_mb": vram_used,
                            "vram_total_mb": vram_total,
                            "vram_percent": vram_percent,
                            "gpu_util_percent": int(parts[5]),
                            "clock_mhz": int(parts[6]),
                            "throttle_status": parts[7] if parts[7] != "0x0000000000000000" else "None"
                        }

        except Exception as e:
            self.logger.debug(f"Failed to get GPU stats: {e}")

        return {
            "temperature_c": 0, "power_draw_w": 0.0,
            "vram_used_mb": 0, "vram_total_mb": self.gpu_total_vram,
            "vram_percent": 0, "gpu_util_percent": 0,
            "clock_mhz": 0, "throttle_status": "Unknown"
        }

    def _get_worker_vram(self, pid: int) -> int:
        """Get VRAM usage for a specific worker PID via nvidia-smi."""
        try:
            result = subprocess.run([
                'nvidia-smi',
                '--query-compute-apps=pid,used_memory',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=5)

            for line in result.stdout.strip().split('\n'):
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 2 and int(parts[0]) == pid:
                    return int(parts[1])
        except Exception:
            pass
        return 0

    def _get_cpu_temp(self) -> Optional[int]:
        """Get the highest CPU temperature reading. Returns temp in C or None."""
        try:
            temps = scan_cpu_temps()
            if temps:
                return max(t["temp_c"] for t in temps)
        except Exception:
            pass
        return None

    def _is_resource_constrained(self, gpu_stats: Dict) -> tuple:
        """Check if GPU resources are constrained. Returns (bool, reasons)."""
        constrained = False
        reasons = []

        gpu_temp = gpu_stats["temperature_c"]
        gpu_warn = self.resource_limits.get("gpu_temp_warning_c", self.resource_limits["max_temp_c"])
        if gpu_temp >= gpu_warn:
            constrained = True
            reasons.append(f"GPU temp {gpu_temp}C >= {gpu_warn}C")

        cpu_temp = self._get_cpu_temp()
        cpu_warn = self.resource_limits.get("cpu_temp_warning_c", 80)
        if cpu_temp is not None and cpu_temp >= cpu_warn:
            constrained = True
            reasons.append(f"CPU temp {cpu_temp}C >= {cpu_warn}C")

        if gpu_stats["vram_percent"] >= self.resource_limits["max_vram_percent"]:
            constrained = True
            reasons.append(f"VRAM {gpu_stats['vram_percent']}% >= {self.resource_limits['max_vram_percent']}%")

        if gpu_stats["power_draw_w"] >= self.resource_limits["max_power_w"]:
            constrained = True
            reasons.append(f"power {gpu_stats['power_draw_w']}W >= {self.resource_limits['max_power_w']}W")

        return constrained, reasons

    def _check_thermal_safety(self, gpu_stats: Dict) -> bool:
        """Check for critical temperatures. Returns True if safe, False if shutdown needed.

        At CRITICAL temps, ALL workers are paused including LLM (emergency override).
        This is the only case where in-flight LLM inference is interrupted.
        """
        gpu_temp = gpu_stats["temperature_c"]
        gpu_crit = self.resource_limits.get("gpu_temp_critical_c", 90)
        cpu_temp = self._get_cpu_temp()
        cpu_crit = self.resource_limits.get("cpu_temp_critical_c", 95)

        critical_reasons = []
        if gpu_temp >= gpu_crit:
            critical_reasons.append(f"GPU {gpu_temp}C >= {gpu_crit}C CRITICAL")
        if cpu_temp is not None and cpu_temp >= cpu_crit:
            critical_reasons.append(f"CPU {cpu_temp}C >= {cpu_crit}C CRITICAL")

        if not critical_reasons:
            return True

        active_task_ids = [
            info["task"]["task_id"][:8] for info in self.active_workers.values()
        ]

        # CRITICAL: Emergency pause ALL workers including LLM (skip_llm=False)
        paused = self._pause_worker_processes(skip_llm=False)

        self.last_thermal_event = {
            "timestamp": datetime.now().isoformat(),
            "event": "critical_shutdown",
            "reasons": critical_reasons,
            "active_task_ids": active_task_ids,
            "emergency_paused": paused,
        }
        self.logger.error(
            f"THERMAL SAFETY SHUTDOWN: {', '.join(critical_reasons)} - "
            f"emergency paused {paused} workers (including LLM), stopping agent. "
            f"TASKS_STALLED_TEMP: {active_task_ids or ['none']}"
        )
        self.running = False
        return False

    def _pause_worker_processes(self, skip_llm: bool = True):
        """Pause active worker subprocesses (best-effort).

        Args:
            skip_llm: If True (default), skip pausing LLM workers to avoid
                      HTTP timeout failures from SIGSTOP during inference.
                      LLM workers should drain naturally via claim throttling.
        """
        paused = 0
        skipped_llm = 0
        for worker_id, info in self.active_workers.items():
            proc = info["process"]
            task_class = info["task"].get("task_class", "cpu")

            # Skip LLM workers to avoid breaking in-flight inference
            if skip_llm and task_class == "llm":
                skipped_llm += 1
                self.logger.info(
                    f"THERMAL_SKIP_LLM_WORKER: allowing {worker_id} to drain "
                    f"task={info['task']['task_id'][:8]}"
                )
                continue

            if proc.poll() is None and not info.get("paused", False):
                try:
                    os.kill(proc.pid, signal.SIGSTOP)
                    info["paused"] = True
                    paused += 1
                    self.logger.warning(
                        f"THERMAL_PAUSE_WORKER: paused {worker_id} pid={proc.pid} "
                        f"task={info['task']['task_id'][:8]} class={task_class}"
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to pause worker {worker_id}: {e}")

        if skipped_llm > 0:
            self.logger.info(f"THERMAL_DRAIN: {skipped_llm} LLM worker(s) draining naturally")
        return paused

    def _resume_worker_processes(self):
        """Resume all paused worker subprocesses (best-effort)."""
        resumed = 0
        for worker_id, info in self.active_workers.items():
            proc = info["process"]
            if proc.poll() is None and info.get("paused", False):
                try:
                    os.kill(proc.pid, signal.SIGCONT)
                    info["paused"] = False
                    resumed += 1
                    self.logger.info(
                        f"THERMAL_RESUME_WORKER: resumed {worker_id} pid={proc.pid} "
                        f"task={info['task']['task_id'][:8]}"
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to resume worker {worker_id}: {e}")
        return resumed

    def _has_active_llm_workers(self) -> bool:
        """Check if any active workers are running LLM tasks."""
        for info in self.active_workers.values():
            if info["task"].get("task_class") == "llm":
                proc = info["process"]
                if proc.poll() is None:  # Still running
                    return True
        return False

    def _temps_below_resume_threshold(self, gpu_stats: Dict) -> bool:
        """Require temps below warning-minus-margin to avoid pause flapping."""
        gpu_warn = self.resource_limits.get("gpu_temp_warning_c", self.resource_limits["max_temp_c"])
        cpu_warn = self.resource_limits.get("cpu_temp_warning_c", 80)
        gpu_limit = max(0, gpu_warn - self.thermal_resume_margin_c)
        cpu_limit = max(0, cpu_warn - self.thermal_resume_margin_c)

        gpu_ok = gpu_stats["temperature_c"] < gpu_limit
        cpu_temp = self._get_cpu_temp()
        cpu_ok = True if cpu_temp is None else (cpu_temp < cpu_limit)
        return gpu_ok and cpu_ok

    def _update_thermal_pause_state(self, gpu_stats: Dict):
        """Manage warning-level thermal pause with exponential backoff and logs.

        Also tracks thermal incidents for brain-level recovery coordination.
        An incident starts when CPU temp exceeds warning threshold and is sustained.
        """
        constrained, reasons = self._is_resource_constrained(gpu_stats)
        now = time.time()
        active_task_ids = [info["task"]["task_id"][:8] for info in self.active_workers.values()]

        # Check for CPU overheat specifically (incident trigger)
        cpu_temp = self._get_cpu_temp()
        cpu_warn = self.resource_limits.get("cpu_temp_warning_c", 80)
        cpu_overheated = cpu_temp is not None and cpu_temp >= cpu_warn

        # Track thermal incident state
        if cpu_overheated:
            if self.thermal_overheat_started_at is None:
                # Start new incident
                self.thermal_overheat_started_at = now
                self.thermal_overheat_incident_id = str(uuid.uuid4())[:8]
                self.thermal_recovery_reset_count = 0
                self.thermal_overheat_sustained_seconds = 0
                self.logger.warning(
                    f"THERMAL_INCIDENT_START: incident_id={self.thermal_overheat_incident_id} "
                    f"cpu_temp={cpu_temp}C threshold={cpu_warn}C"
                )
            else:
                # Update sustained duration
                self.thermal_overheat_sustained_seconds = int(now - self.thermal_overheat_started_at)
        else:
            if self.thermal_overheat_started_at is not None:
                # Clear incident
                incident_id = self.thermal_overheat_incident_id
                duration = self.thermal_overheat_sustained_seconds
                reset_count = self.thermal_recovery_reset_count
                self.logger.info(
                    f"THERMAL_INCIDENT_CLEARED: incident_id={incident_id} "
                    f"duration={duration}s resets_issued={reset_count} cpu_temp={cpu_temp}C"
                )
                self.thermal_overheat_started_at = None
                self.thermal_overheat_sustained_seconds = 0
                self.thermal_overheat_incident_id = None
                self.thermal_recovery_reset_count = 0

        if constrained:
            self.thermal_pause_reasons = reasons.copy()
            if not self.thermal_pause_active:
                self.thermal_pause_active = True
                self.thermal_pause_attempts = 1
                self.thermal_pause_current_seconds = self.thermal_pause_initial_seconds
                self.thermal_pause_until = now + self.thermal_pause_current_seconds
                paused = self._pause_worker_processes()
                self.last_thermal_event = {
                    "timestamp": datetime.now().isoformat(),
                    "event": "pause_entered",
                    "reasons": reasons.copy(),
                    "pause_seconds": self.thermal_pause_current_seconds,
                    "pause_attempt": self.thermal_pause_attempts,
                    "active_task_ids": active_task_ids,
                }
                self.logger.warning(
                    f"THERMAL_PAUSE_ENTER: reasons={reasons} pause={self.thermal_pause_current_seconds}s "
                    f"paused_workers={paused} active_task_ids={active_task_ids or ['none']}"
                )
            elif now >= self.thermal_pause_until:
                self.thermal_pause_attempts += 1
                next_pause = int(self.thermal_pause_current_seconds * self.thermal_pause_backoff_factor)
                self.thermal_pause_current_seconds = min(self.thermal_pause_max_seconds, max(1, next_pause))
                self.thermal_pause_until = now + self.thermal_pause_current_seconds
                paused = self._pause_worker_processes()
                self.last_thermal_event = {
                    "timestamp": datetime.now().isoformat(),
                    "event": "pause_extended",
                    "reasons": reasons.copy(),
                    "pause_seconds": self.thermal_pause_current_seconds,
                    "pause_attempt": self.thermal_pause_attempts,
                    "active_task_ids": active_task_ids,
                }
                self.logger.warning(
                    f"THERMAL_PAUSE_EXTEND: reasons={reasons} pause={self.thermal_pause_current_seconds}s "
                    f"attempt={self.thermal_pause_attempts} paused_workers={paused} "
                    f"active_task_ids={active_task_ids or ['none']}"
                )
            return

        if self.thermal_pause_active and now >= self.thermal_pause_until:
            if self._temps_below_resume_threshold(gpu_stats):
                resumed = self._resume_worker_processes()
                self.thermal_pause_active = False
                self.thermal_pause_reasons = []
                self.thermal_pause_until = 0.0
                self.thermal_pause_current_seconds = 0
                self.thermal_pause_attempts = 0
                self.last_thermal_event = {
                    "timestamp": datetime.now().isoformat(),
                    "event": "pause_cleared",
                    "reasons": [],
                    "active_task_ids": active_task_ids,
                }
                self.logger.info(
                    f"THERMAL_PAUSE_EXIT: resumed_workers={resumed} active_task_ids={active_task_ids or ['none']}"
                )
            else:
                self.thermal_pause_attempts += 1
                next_pause = int(self.thermal_pause_current_seconds * self.thermal_pause_backoff_factor)
                self.thermal_pause_current_seconds = min(self.thermal_pause_max_seconds, max(1, next_pause))
                self.thermal_pause_until = now + self.thermal_pause_current_seconds
                self.last_thermal_event = {
                    "timestamp": datetime.now().isoformat(),
                    "event": "pause_hold_hot",
                    "reasons": self.thermal_pause_reasons.copy(),
                    "pause_seconds": self.thermal_pause_current_seconds,
                    "pause_attempt": self.thermal_pause_attempts,
                    "active_task_ids": active_task_ids,
                }
                self.logger.warning(
                    f"THERMAL_PAUSE_HOLD: temp recovery threshold not met, "
                    f"extending pause to {self.thermal_pause_current_seconds}s "
                    f"attempt={self.thermal_pause_attempts}"
                )

    def _get_vram_budget(self) -> int:
        """Calculate available VRAM budget for new tasks."""
        total_budget = int(self.gpu_total_vram * VRAM_BUDGET_RATIO)
        available = total_budget - self.claimed_vram
        return max(0, available)

    def _get_task_vram_cost(self, task: Dict) -> int:
        """Get the VRAM cost for a task."""
        task_class = task.get("task_class", "cpu")

        if task_class == "llm":
            return int(self.gpu_total_vram * VRAM_BUDGET_RATIO)  # LLM uses full budget
        elif task_class == "cpu":
            return DEFAULT_CPU_VRAM_COST
        elif task_class == "script":
            return task.get("vram_estimate_mb", DEFAULT_CPU_VRAM_COST)
        elif task_class == "meta":
            return 0  # Meta tasks handled by GPU directly
        else:
            return DEFAULT_CPU_VRAM_COST
