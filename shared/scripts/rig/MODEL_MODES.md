# Named experimental model loads

Orchestration owns the executable load settings. Benchmarking owns the
measurements and receipts that justify a mode. A caller names a model and a
mode; the ordinary rig controller handles single-server modes. The shared-KV
35B service below uses the same rig GPU claims for admission, launches its own
one-model runtime, verifies readiness, and reports a per-chat endpoint.
The gateway and harness should use the returned endpoint and never copy GPU maps, tensor patterns, or KV flags.
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

For ordinary single-server modes, the optional `--lease ID` attaches an existing
manager lease. The shared-KV mode currently uses GPU claims and rejects
`--lease`; `start` never unloads another model or forces a conflicting mode. The existing rig controller
rejects busy cards and ports. Use its normal stop command for an intentional
mode switch. `--detach` runs a tracked rig load job; readiness is visible in
`rig observe`. Without it, `start` waits for readiness.

| Canonical model ID | Mode | Placement | Evidence |
| --- | --- | --- | --- |
| `qwen3.6:35b-a3b-xl` | `single-chat-fast` | Single GPU 0 server, 16K cap, reasoning off | Benchmarking model benchmark reference |
| `qwen3.6:35b-a3b-xl` | `gpu0-main-q4` | GPU0 weights and main chat, 262K Q4_0 KV cap | Benchmarking six-chat replay |
| `nemotron-3.5-bolt:74b-a9b` | `single-chat-fast` | GPU 0 chat, routed experts across GPUs 1–4, 262K cap, F16 KV | Benchmarking `results/runtime_experiments/nemo_chat3090_20261109/` |
| `qwen3-coder-next:80b-a3b` | `single-chat-fast` | GPU 0 chat, 30 expert layers across GPUs 1–5, 262K cap, F16 KV | Benchmarking `results/runtime_experiments/coder_next_chat_20261111/` |

## Preferred 35B placement and attached chats

The preferred experimental capacity layout is one Qwen3.6 35B-A3B XL Q4_K_XL
weight load on GPU0, one Q4_0 main KV context on GPU0, and up to five Q4_0
worker KV contexts, one on each of GPUs1?5. The [six-chat replay](https://github.com/bigbertharig/benchmarking/blob/2bc9fda/results/runtime_experiments/model_mode_smokes_20261111/README.md)
ran all six at 16K, 32K, and 50K occupied depth with 262K reservations. GPU0
peaked at 23,567 MiB of 24,564 MiB, so this fit has little GPU0 margin.

The model library has **two request keys**:

| Model | Mode | Type | Requires |
| --- | --- | --- | --- |
| `qwen3.6:35b-a3b-xl` | `gpu0-main-q4` | Persistent shared-weight model and main chat on GPU0 | GPU0 free; patched KV runtime built |
| `qwen3.6:35b-a3b-xl` | `gpu1060-chat-q4` | Attach one independent worker chat to a free 1060 | Ready `gpu0-main-q4` parent |

Build the private runtime once on the rig, then use the named commands:

```bash
/home/bryan/llm_orchestration/shared/scripts/rig/build_kv_chat_server.sh
python3 /home/bryan/llm_orchestration/shared/scripts/rig/model_modes.py start qwen3.6:35b-a3b-xl gpu0-main-q4 --port 11524
python3 /home/bryan/llm_orchestration/shared/scripts/rig/model_modes.py start qwen3.6:35b-a3b-xl gpu1060-chat-q4 --chat-id worker-a
python3 /home/bryan/llm_orchestration/shared/scripts/rig/model_modes.py turn qwen3.6:35b-a3b-xl gpu1060-chat-q4 --chat-id worker-a --message 'What is 2+2?' --max-tokens 32
python3 /home/bryan/llm_orchestration/shared/scripts/rig/model_modes.py stop qwen3.6:35b-a3b-xl gpu1060-chat-q4 --chat-id worker-a
python3 /home/bryan/llm_orchestration/shared/scripts/rig/model_modes.py stop qwen3.6:35b-a3b-xl gpu0-main-q4
```

`--gpu 1` through `--gpu 5` selects a worker explicitly; omitting it chooses
the first free 1060. The parent command holds a GPU0 claim and starts one
container with one model instance. Each attach command claims its 1060 before
creating a new KV context in that process; detach frees the context and claim.
`status` reports the live chats and their cards. `rig status` also reports the
shared-KV service beside its ordinary mode state, and a global `rig stop`
cleans up the service and its claims. The returned turn route is
loopback on the rig; the Gateway/harness must invoke the controller or use its
existing rig tunnel to reach that route. The parent stays ready while children
exist, and stopping it releases all child claims.

This is an experimental inference service around the private patched llama.cpp
build. It supports text turns with greedy decoding and per-chat Q4_0 KV. It is
not yet an OpenAI-compatible route or a published Gateway offer; those adapters
must use the returned chat ID and route, and handle model/worker admission
through this controller. It has no chat checkpoint durability across parent
restart. Long-context turn quality and sustained multi-turn speed still need
verification. Never substitute a second `rig start remote`: that would load
another copy of the weights.

These are experimental single-server modes on the current six-card rig and
model files. The Nemo and Coder layout measurements used a private llama.cpp
probe; the named HTTP server recipes still need matched quality and speed
verification before general gateway offers. The Nemo two-chat placement still needs a corresponding shared-weight
runtime. The 35B attached-chat mode uses the separate experimental service
above, not the ordinary llama-server. The CLI rejects unknown model/mode pairs instead of silently
substituting another placement.

The current interface is deliberately small. It gives experiments one load
command and one place to change a placement. Promoting these modes to routine
gateway requests also needs the manager admission and lease contract in
`/home/bryan/llm-gateway/docs/MODEL_LIBRARY_HANDOFF.md`.

The first 35B XL HTTP smoke returned empty answer content on many short
requests because thinking filled the generation budget. Its mode explicitly
passes `--reasoning off`; the corrected smoke receipt is recorded in the
benchmarking repo. A stage exit code alone is not a quality gate.
