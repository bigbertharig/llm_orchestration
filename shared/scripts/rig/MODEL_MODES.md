# Named experimental model loads

Orchestration owns the executable load settings. Benchmarking owns the
measurements and receipts that justify a mode. A caller names a model and a
mode; the rig controller applies admission, launches the runtime, verifies
readiness, and reports the endpoint. The gateway and harness should use the
returned endpoint and never copy GPU maps, tensor patterns, or KV flags.
The request key is **canonical registry model ID + mode**. The mode describes
the requested capability; context capacity and hardware placement are resolved
by that pair. `list --json` exposes the same two columns for callers.

On the rig:

```bash
python3 /home/bryan/llm_orchestration/shared/scripts/rig/model_modes.py list
python3 /home/bryan/llm_orchestration/shared/scripts/rig/model_modes.py show qwen3-coder-next:80b-a3b single-chat-fast
python3 /home/bryan/llm_orchestration/shared/scripts/rig/model_modes.py start qwen3-coder-next:80b-a3b single-chat-fast --port 11522 --dry-run
python3 /home/bryan/llm_orchestration/shared/scripts/rig/model_modes.py start qwen3-coder-next:80b-a3b single-chat-fast --port 11522 --detach
rig observe
```

The optional `--lease ID` attaches an existing manager lease. `start` never
unloads another model or forces a conflicting mode. The existing rig controller
rejects busy cards and ports. Use its normal stop command for an intentional
mode switch. `--detach` runs a tracked rig load job; readiness is visible in
`rig observe`. Without it, `start` waits for readiness.

| Canonical model ID | Mode | Placement | Evidence |
| --- | --- | --- | --- |
| `qwen3.6:35b-a3b-xl` | `single-chat-fast` | Single GPU 0 server, 16K cap, reasoning off | Benchmarking model benchmark reference |
| `qwen3.6:35b-a3b-xl` | `gpu0-main-q4` | GPU0 weights and main chat, 262K Q4_0 KV cap | Benchmarking six-chat replay |
| `nemotron-3.5-bolt:74b-a9b` | `single-chat-fast` | GPU 0 chat, routed experts across GPUs 1–4, 262K cap, F16 KV | Benchmarking `results/runtime_experiments/nemo_chat3090_20261109/` |
| `qwen3-coder-next:80b-a3b` | `single-chat-fast` | GPU 0 chat, 30 expert layers across GPUs 1–5, 262K cap, F16 KV | Benchmarking `results/runtime_experiments/coder_next_chat_20261111/` |

## Preferred 35B placement and attached-chat dependency

The preferred experimental capacity layout is one Qwen3.6 35B-A3B XL Q4_K_XL
weight load on GPU0, one Q4_0 main KV context on GPU0, and up to five Q4_0
worker KV contexts, one on each of GPUs1?5. The [six-chat replay](https://github.com/bigbertharig/benchmarking/blob/2bc9fda/results/runtime_experiments/model_mode_smokes_20261111/README.md)
ran all six at 16K, 32K, and 50K occupied depth with 262K reservations. GPU0
peaked at 23,567 MiB of 24,564 MiB, so this fit has little GPU0 margin.
It measured short 64-token windows; admission and longer turns still need
runtime validation.

The model library has **two distinct request keys**:

| Model | Mode | Type | Requires | Runtime state |
| --- | --- | --- | --- | --- |
| `qwen3.6:35b-a3b-xl` | `gpu0-main-q4` | Model + main chat | GPU0 | Executable single-server load, 262K cap, Q4_0 KV |
| `qwen3.6:35b-a3b-xl` | `gpu1060-chat-q4` | Add one worker chat | Ready `gpu0-main-q4` instance, then one free GPU1?5 | Catalogued dependency; attach endpoint not implemented |

Inspect the two entries with `model_modes.py show MODEL MODE` or
`model_modes.py list --json`. The main load command is:

```bash
python3 /home/bryan/llm_orchestration/shared/scripts/rig/model_modes.py start qwen3.6:35b-a3b-xl gpu0-main-q4 --port 11524 --detach
```

`start ... gpu1060-chat-q4` currently returns a clear error. The private
benchmark runner proves device placement, but its replay is not an HTTP chat
service and cannot attach a live conversation to the existing llama-server.
A second `rig start remote` would load a second model process, so it is not a
substitute. Gateway and harness can discover the dependent entry but must not
advertise it as ready or request worker chat admission until a shared-model
runtime supports per-chat KV device selection and returns a separate chat route.
The parent instance must remain ready for the entire worker lifetime; worker
card leases and per-chat teardown belong to that future attach operation.

These are experimental single-server modes on the current six-card rig and
model files. The Nemo and Coder layout measurements used a private llama.cpp
probe; the named HTTP server recipes still need matched quality and speed
verification before general gateway offers. The 35B five-worker and Nemo
two-chat placements need independent per-chat KV device selection; the current
HTTP server cannot express them. They remain in benchmarking until that runtime
capability exists. The CLI rejects unknown model/mode pairs instead of silently
substituting another placement.

The current interface is deliberately small. It gives experiments one load
command and one place to change a placement. Promoting these modes to routine
gateway requests also needs the manager admission and lease contract in
`/home/bryan/llm-gateway/docs/MODEL_LIBRARY_HANDOFF.md`.

The first 35B XL HTTP smoke returned empty answer content on many short
requests because thinking filled the generation budget. Its mode explicitly
passes `--reasoning off`; the corrected smoke receipt is recorded in the
benchmarking repo. A stage exit code alone is not a quality gate.
