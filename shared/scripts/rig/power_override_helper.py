#!/usr/bin/python3
"""Installed root-owned helper for one scoped GPU power cap change.

Install a reviewed copy at /usr/local/libexec/rig-power-override. This file imports
only the Python standard library and accepts no shell commands or filesystem paths.
"""
import json
import math
import os
from pathlib import Path
import subprocess
import sys

OVERRIDE = Path('/run/rig/power_override.json')

def run(gpu_text, value_text, target_text):
    if os.geteuid() != 0:
        raise ValueError('root required')
    if not gpu_text.isdecimal() or len(gpu_text) > 2:
        raise ValueError('invalid GPU index')
    gpu = int(gpu_text)
    query = subprocess.run(['nvidia-smi','--query-gpu=index,power.limit,power.min_limit,power.max_limit',
                            '--format=csv,noheader,nounits'],check=True,capture_output=True,text=True,timeout=10)
    cards = {}
    for line in query.stdout.splitlines():
        fields = [x.strip() for x in line.split(',')]
        if len(fields) == 4 and fields[0].isdecimal():
            cards[int(fields[0])] = tuple(float(x) for x in fields[1:])
    if gpu not in cards:
        raise ValueError('GPU absent')
    current, minimum, maximum = cards[gpu]
    def watts(text):
        value = float(text)
        if not math.isfinite(value) or value < minimum or value > maximum:
            raise ValueError('watts outside hardware bounds')
        return value
    target = watts(target_text)
    value = None if value_text == 'restore' else watts(value_text)
    if value is not None and value != target:
        raise ValueError('override and applied cap disagree')
    old_bytes = OVERRIDE.read_bytes() if OVERRIDE.is_file() else None
    doc = json.loads(old_bytes) if old_bytes else {}
    values = doc.setdefault('by_index', {})
    previous = values.get(str(gpu))
    if value is None: values.pop(str(gpu), None)
    else: values[str(gpu)] = value
    OVERRIDE.parent.mkdir(parents=True,exist_ok=True)
    tmp = OVERRIDE.with_name(OVERRIDE.name+'.new')
    try:
        with tmp.open('w') as out:
            json.dump(doc,out);out.flush();os.fsync(out.fileno())
        os.replace(tmp,OVERRIDE)
        subprocess.run(['nvidia-smi','-i',str(gpu),'-pl',str(target)],
                       check=True,capture_output=True,text=True,timeout=15)
    except BaseException:
        if old_bytes is None: OVERRIDE.unlink(missing_ok=True)
        else:
            with tmp.open('wb') as out:
                out.write(old_bytes);out.flush();os.fsync(out.fileno())
            os.replace(tmp,OVERRIDE)
        try: subprocess.run(['nvidia-smi','-i',str(gpu),'-pl',str(current)],
                            check=True,capture_output=True,text=True,timeout=15)
        except Exception: pass
        raise
    return {'gpu_id':gpu,'previous_override_w':previous,'override_w':value,'applied_w':target}

if __name__ == '__main__':
    try:
        if len(sys.argv) != 4: raise ValueError('GPU, override and applied watts required')
        print(json.dumps(run(*sys.argv[1:])))
    except Exception as exc:
        print(str(exc),file=sys.stderr)
        raise SystemExit(1)
