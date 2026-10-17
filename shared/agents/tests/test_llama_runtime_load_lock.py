#!/usr/bin/env python3
"""run_runtime.sh must load one model at a time, no matter who calls it.

Uses fake docker and curl binaries on PATH: a fake "docker run" marks its port as
loading and becomes healthy after LOAD_S seconds. Two launches started together must
not overlap: the second container may only start after the first answers /health.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import tempfile
import time
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "llama_runtime" / "run_runtime.sh"
LOAD_S = 1.5

FAKE_DOCKER = r"""#!/bin/bash
state="${FAKE_STATE:?}"
cmd="$1"; shift
case "$cmd" in
  run)
    name=""; port=""; model=""
    while [ $# -gt 0 ]; do
      case "$1" in
        --name) name="$2"; shift 2 ;;
        --port) port="$2"; shift 2 ;;
        --model) model="$2"; shift 2 ;;
        *) shift ;;
      esac
    done
    echo "$(date +%s.%N) start $name" >> "$state/events"
    echo "$port" > "$state/container_$name"
    [ -f "$state/serve_$port" ] || echo "$model" > "$state/serve_$port"
    if [ -f "$state/fail_$name" ]; then
      echo exited > "$state/exited_$name"
    else
      ( sleep "${FAKE_LOAD_S:?}"; touch "$state/ready_$port";
        echo "$(date +%s.%N) ready $name" >> "$state/events" ) >/dev/null 2>&1 &
    fi
    echo "id-$name"
    ;;
  ps)
    for f in "$state"/container_*; do
      [ -e "$f" ] || continue
      name="${f##*/container_}"
      [ -f "$state/exited_$name" ] && continue
      echo "id-$name http://127.0.0.1:$(cat "$f")/health"
    done
    ;;
  inspect)
    name="${@: -1}"
    if [ -f "$state/exited_$name" ]; then echo false; else echo true; fi
    ;;
  rm)
    name="${@: -1}"
    rm -f "$state/container_$name"
    echo "$(date +%s.%N) removed $name" >> "$state/events"
    ;;
  logs) echo "fake logs" ;;
esac
exit 0
"""

FAKE_CURL = r"""#!/bin/bash
url="${@: -1}"
port="${url##*:}"; port="${port%%/*}"
[ -f "${FAKE_STATE:?}/ready_$port" ] || exit 7
case "$url" in
  */v1/models) printf '{"data": [{"id": "%s"}]}' "$(cat "$FAKE_STATE/serve_$port")" ;;
esac
exit 0
"""


@unittest.skipUnless(shutil.which("flock") and shutil.which("bash"), "needs flock and bash")
class RuntimeLoadLockTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.bin = self.tmp / "bin"
        self.state = self.tmp / "state"
        self.bin.mkdir()
        self.state.mkdir()
        for name, body in (("docker", FAKE_DOCKER), ("curl", FAKE_CURL)):
            path = self.bin / name
            path.write_text(body, encoding="utf-8", newline="\n")
            path.chmod(path.stat().st_mode | stat.S_IEXEC)
        self.model = self.tmp / "model.gguf"
        self.model.write_text("stub", encoding="utf-8")
        self.env = dict(
            os.environ,
            PATH=f"{self.bin}{os.pathsep}{os.environ['PATH']}",
            FAKE_STATE=str(self.state),
            FAKE_LOAD_S=str(LOAD_S),
            LLAMA_LOAD_LOCK=str(self.tmp / "load.lock"),
            LLAMA_LOAD_SETTLE_S="0",
            LLAMA_LOAD_READY_TIMEOUT_S="20",
        )

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _launch(self, name: str, port: int) -> subprocess.Popen:
        return subprocess.Popen(
            ["bash", str(SCRIPT), "--name", name, "--model", str(self.model),
             "--port", str(port), "--gpus", "device=1", "--image", "llama-runtime:test"],
            env=self.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )

    def _events(self) -> list[tuple[float, str, str]]:
        rows = []
        for line in (self.state / "events").read_text(encoding="utf-8").splitlines():
            ts, kind, name = line.split()
            rows.append((float(ts), kind, name))
        return sorted(rows)

    def test_simultaneous_launches_load_one_at_a_time(self):
        procs = [self._launch("rt-a", 18001), self._launch("rt-b", 18002)]
        outs = [p.communicate(timeout=30) for p in procs]
        for proc, (out, err) in zip(procs, outs):
            self.assertEqual(proc.returncode, 0, err)
            self.assertTrue(out.strip().startswith("id-rt-"), out)

        events = self._events()
        kinds = [kind for _ts, kind, _name in events]
        # Strictly start, ready, start, ready: never two loads in flight.
        self.assertEqual(kinds, ["start", "ready", "start", "ready"], events)
        self.assertEqual(events[0][2], events[1][2])

    def test_waits_for_a_runtime_still_loading_without_the_lock(self):
        # A container left loading by a killed launcher (lock already released).
        (self.state / "container_orphan").write_text("18009", encoding="utf-8")
        started = time.time()

        def finish_orphan():
            time.sleep(LOAD_S)
            (self.state / "ready_18009").touch()

        import threading
        threading.Thread(target=finish_orphan, daemon=True).start()
        proc = self._launch("rt-c", 18003)
        _out, err = proc.communicate(timeout=30)
        self.assertEqual(proc.returncode, 0, err)
        first_start = next(ts for ts, kind, _n in self._events() if kind == "start")
        self.assertGreaterEqual(first_start - started, LOAD_S - 0.2)

    def test_failed_load_is_removed_before_the_lock_is_released(self):
        (self.state / "fail_rt-bad").touch()
        proc = self._launch("rt-bad", 18004)
        _out, err = proc.communicate(timeout=30)
        self.assertEqual(proc.returncode, 70, err)
        kinds = [kind for _ts, kind, name in self._events() if name == "rt-bad"]
        self.assertEqual(kinds, ["start", "removed"])

    def test_refuses_a_port_another_runtime_is_serving(self):
        # Another model already answers on the port (left over from an earlier test).
        (self.state / "ready_18010").touch()
        (self.state / "serve_18010").write_text("/models/other.gguf", encoding="utf-8")
        proc = self._launch("rt-d", 18010)
        _out, err = proc.communicate(timeout=30)
        self.assertEqual(proc.returncode, 75, err)
        self.assertIn("already serving other.gguf", err)
        self.assertFalse((self.state / "events").exists())

    def test_wrong_model_on_the_port_fails_the_load(self):
        # The probe is answered, but by a server reporting a different GGUF.
        (self.state / "serve_18011").write_text("/models/other.gguf", encoding="utf-8")
        proc = self._launch("rt-e", 18011)
        _out, err = proc.communicate(timeout=30)
        self.assertEqual(proc.returncode, 70, err)
        self.assertIn("expected 'model.gguf'", err)
        kinds = [kind for _ts, kind, name in self._events() if name == "rt-e"]
        self.assertEqual(kinds[0], "start")
        self.assertIn("removed", kinds)


if __name__ == "__main__":
    unittest.main()
