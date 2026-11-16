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
`instance_id`, `port`, and `chats`. `rig shared-kv status` also includes `claims`.
The `instance_id` changes on every service restart; clients should bind their
conversation state to it. Turns accept plain text and return JSON with content,
GPU, token counters, and timing. The server does not expose native tool calls.

The model file and override table are the hotset and benchmark paths recorded
in `kv_chat_control.py`. `build_kv_chat_server.sh` documents the base build;
the controller uses a separately compiled `llama-kv-chat-server-v2` binary in
that build directory. A 600-second Nemo startup window covers the cold weight
load and main F16 context creation. The 35B service continues to use its
existing mode and can be selected explicitly with `qwen3.6:35b-a3b-xl`.

The benchmark baseline is
`/mnt/shared/plans/shoulders/benchmarking/MOE_CHAT_BASELINE_RELOAD_2026-11-09.md`.
The benchmark tested two independent chats through 50K occupied tokens; it
did not test two chats at 250K. The service validation receipt is in
`NEMO_SHARED_CHAT_RECEIPT_2026-11-16.md`.
