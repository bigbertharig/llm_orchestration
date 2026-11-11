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
| `qwen3.6:35b-a3b-xl` | `single-chat-fast` | Single GPU 0 server, 16K cap, ordinary baseline | Benchmarking model benchmark reference |
| `nemotron-3.5-bolt:74b-a9b` | `single-chat-fast` | GPU 0 chat, routed experts across GPUs 1–4, 262K cap, F16 KV | Benchmarking `results/runtime_experiments/nemo_chat3090_20261109/` |
| `qwen3-coder-next:80b-a3b` | `single-chat-fast` | GPU 0 chat, 30 expert layers across GPUs 1–5, 262K cap, F16 KV | Benchmarking `results/runtime_experiments/coder_next_chat_20261111/` |

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
