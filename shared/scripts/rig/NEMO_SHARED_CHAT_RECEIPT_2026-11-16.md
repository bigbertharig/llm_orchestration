# Nemotron shared chat service validation — 2026-11-16

Rig checkout: `/home/bryan/llm_orchestration`. Model:
`nemotron-3.5-bolt:74b-a9b`. Service instance:
`fe69182ae1e54d88b26f7b6c023993a7`. Mode:
`gpu0-main-f16-experts1234`, port `11525`. At receipt time the service was
left running for gateway validation.

## Build and startup

- `python3 -m py_compile shared/scripts/rig/kv_chat_control.py shared/scripts/rig/rig.py`: exit 0.
- New `llama-kv-chat-server-v2` compiled with the repo's CUDA build recipe: exit 0.
- Existing 35B service had only `main`, zero used tokens, and one GPU0 claim.
  It was stopped through `rig shared-kv stop --model qwen3.6:35b-a3b-xl`.
- First Nemo cold load placed 20,558.75 MiB model buffer on GPU0 and roughly
  3.9–4.2 GiB on each of GPUs1-4. It exceeded the inherited 120-second
  readiness window; the controller stopped the container and released its
  claims. A retry with a 600-second Nemo window succeeded. No concurrent
  35B service was left running.
- `rig shared-kv start --model nemotron-3.5-bolt:74b-a9b`: exit 0, ready,
  main chat GPU0. `rig shared-kv attach --model nemotron-3.5-bolt:74b-a9b
  --chat-id child5 --gpu 5`: exit 0, 262,144 context cap, same instance.

## HTTP chat checks

- Distinct initial 16-token turns succeeded: main GPU0 at 30 used tokens,
  child5 GPU5 at 31. Both reported independent content and token counters.
- Concurrent short HTTP turns succeeded: main GPU0 at 64 used tokens, child5
  GPU5 at 66; each returned 16 tokens.
- `rig shared-kv attach --model nemotron-3.5-bolt:74b-a9b --chat-id second
  --gpu 4` exited 2 with `GPU 4 is not an eligible worker card; detected [5]`.
  GPU4 is reserved for expert weights.
- A deterministic 61,999-byte message filled main to 37,280 used tokens in
  340.12 seconds and child5 to 37,282 in 435.85 seconds. Both contexts were
  occupied simultaneously. Concurrent 16-token turns then succeeded: main
  GPU0 at 37,314 used tokens and 33.54 decode tok/s; child5 GPU5 at 37,317
  and 28.73 decode tok/s. The probe exited 0. Raw JSON lines are in
  `NEMO_SHARED_CHAT_PROBE_2026-11-16.jsonl`.

`rig shared-kv status` and `rig status --json` include the same model, mode,
port, instance ID, and both chat rows. Claims are `nemo74-kv-main` on
GPUs0-4 and `nemo74-kv-chat-child5` on GPU5. The 262,144 positions are
reserved context caps; this service test validated about 37K occupied tokens
per chat. Two chats near 250K remain untested. Restart behavior was verified
by code inspection (`uuid.uuid4().hex` on successful start); the live Nemo
instance was not restarted because that would discard the tested chats.
