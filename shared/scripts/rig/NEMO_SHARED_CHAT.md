# Nemotron shared-weight chat mode

The rig owns one shared-weight chat service at a time. The explicit model ID is
`nemotron-3.5-bolt:74b-a9b`. Its proven placement loads model weights on GPU0,
places whole experts on GPUs1-4 using the benchmark tensor override table,
puts the main F16 KV context on GPU0, and permits one attached F16 KV context
on GPU5. Each chat reserves 262,144 context positions. The placement is fixed;
there is no second worker slot in this mode.

```text
rig shared-kv start --model nemotron-3.5-bolt:74b-a9b
rig shared-kv status --model nemotron-3.5-bolt:74b-a9b
rig shared-kv attach --model nemotron-3.5-bolt:74b-a9b --chat-id child5 --gpu 5
rig shared-kv turn --model nemotron-3.5-bolt:74b-a9b --chat-id child5 --message TEXT --max-tokens 128
rig shared-kv detach --model nemotron-3.5-bolt:74b-a9b --chat-id child5
rig shared-kv stop --model nemotron-3.5-bolt:74b-a9b
```

`rig status --json` exposes `shared_kv_service` with `status`, `model`, `mode`,
`instance_id`, `port`, `chats`, `worker_capacity`, and the authoritative context
`reservations`. It obtains this record from `kv_chat_control.py status`, so the
Gateway and the rig use the same admission calculation. The top-level and
per-chat reservation views report live occupied-token telemetry.
The `instance_id` changes on every service restart; clients should bind their
conversation state to it. `occupied_tokens` is telemetry and never reduces the
262,144-token reservation. Turns accept plain text and return JSON with content,
GPU, token counters, and timing. The server does not expose native tool calls.

The model file and override table are the hotset and benchmark paths recorded
in `kv_chat_control.py`. `build_kv_chat_server.sh` documents the base build and
produces `llama-kv-chat-server-v2`. A 600-second Nemo startup window covers the
cold weight load and main F16 context creation. The 35B service continues to
use its existing mode and can be selected explicitly with
`qwen3.6:35b-a3b-xl`.

The benchmark baseline is
`/mnt/shared/plans/shoulders/benchmarking/MOE_CHAT_BASELINE_RELOAD_2026-11-09.md`.
The benchmark tested two independent chats through 50K occupied tokens; it did
not test two chats at 250K. The service validation receipt is in
`NEMO_SHARED_CHAT_RECEIPT_2026-11-16.md`.

## Same-card stacking boundary

`worker_capacity.placement_scope` is `one_chat_per_worker_gpu` for the current
HTTP service, and `/chats` rejects a second chat on the same card. The
November 7 three-chat Nemotron result did share GPUs4-5, but used the private
b11669 per-context placement runtime, Q4 KV, and a different four-card weight
split. Its 131K chat had only a short occupied prompt. The November 9 Qwen35
placement sweep ran two KVs on GPU0 with weights, but used replay rather than
this HTTP server and measured 16K/32K histories. Neither is an admission proof
for another 262K F16 chat on GPU5 in this production service.

Future stacking admission should require a matching runtime build, model
placement, KV format, per-chat context reservation, and a concurrent occupied
depth test. The controller must then reserve cumulative memory atomically for
all chats sharing a card and retain a card claim until its final chat detaches.
Free VRAM alone cannot prove that the runtime can host multiple contexts on
that card. The present exclusive policy applies across rigs without relying on
fixed card identities; this Nemotron recipe's eligible card remains specific to
the tested six-card placement.

## Live rollout, 2026-11-18

The upgraded controller and server were installed on the rig and the Gateway
was restarted. The shared service loaded with instance
`95764d53fecb44f4bd5b5f3c315b6738`. A main chat on GPU0 and a worker on
GPU5 both generated replies. The worker received a 262,144-token exclusive
reservation receipt, a further attach was rejected with no eligible free GPU,
and detach released the claim. A Gateway `agent-routes/resolve` call then
allocated `child-ad5b8e6905474607` with a matching service instance and
reservation receipt. `local-agent-capabilities` showed one active child and
zero free slots, and `agent-routes/release` released that child. The temporary
test chats were removed; the shared parent remains loaded. A subsequent
end-to-end Gateway test allocated a child from `gpu-rig/shared-nemo-main`,
sent concurrent completions through both routes, received `MAIN` and `WORKER`
from their respective chats, and released the child. Its request, admission,
answers, and release receipt are saved in
`NEMO_SHARED_CHAT_GATEWAY_RECEIPT_2026-11-18.json`.
