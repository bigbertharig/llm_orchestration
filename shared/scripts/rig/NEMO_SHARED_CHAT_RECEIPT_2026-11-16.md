# Nemotron shared chat service validation — 2026-11-16

Rig checkout: `/home/bryan/llm_orchestration`. Model:
`nemotron-3.5-bolt:74b-a9b`. Service instance:
`fe69182ae1e54d88b26f7b6c023993a7`. Mode:
`gpu0-main-f16-experts1234`, port `11525`.

The validated cold load placed the model buffer on GPU0 and whole experts on
GPUs1-4. The main chat used GPU0; one attached worker used GPU5. Both advertised
262,144-token F16 contexts. An attempted GPU4 attachment was correctly rejected
because GPU4 held expert weights.

Distinct initial and concurrent short turns passed. A deterministic fill put
the main chat at 37,280 occupied tokens and the worker at 37,282. Concurrent
16-token turns then completed at 33.54 and 28.73 decode tokens/s respectively,
with independent content and counters. Raw rows are preserved in
`NEMO_SHARED_CHAT_PROBE_2026-11-16.jsonl`.

Claims were `nemo74-kv-main` on GPUs0-4 and `nemo74-kv-chat-child5` on GPU5.
The 262,144 positions are configured reservations; the test validated about
37K occupied tokens per chat. Two chats near 250K, checkpoint persistence, and
long-context answer quality remain untested.
