# Chat roles versus shared-model placement

The Gateway identifies the **main chat**: the controlling conversation that
receives a user's requests and delegated results. The harness runs an **agent
chat** as a bounded delegated conversation with its own context, tool policy,
and receipt. Either chat can use a local model; a cloud main chat has no local
KV context. Model size and GPU identity do not determine either role.

Orchestration owns the **shared model service** (loaded weights and runtime
state), physical placement, context reservations, and admission. It does not
decide which conversation controls work. A headless shared service can have
zero chats while remaining loaded. Its `reservations.main` entry is a legacy
**service parent reservation** for shared weights and parent-side memory; a
zero-context `main` reservation is not a main conversation. Likewise,
`gpu0-main-q4` is a tested service-mode name, not a rule assigning the
controlling chat to GPU0.

For each attach request, Gateway supplies the chosen model/service recipe,
requested context capacity, and chat identity. Orchestration selects an
eligible placement and returns the service instance, chat route, GPU, and
reservation receipt. The service instance and receipt bind future turns and
release. The current controller uses the imported benchmark limits plus live
memory and claim checks: Qwen's main-chat service recipe permits at most six
attached worker chats; its headless worker-pool recipe permits at most sixteen
default-context worker chats, with eleven full-context chats validated in the
benchmark. These are recipe limits and may admit fewer chats on a different
rig or at a different context size.

Benchmarking owns evidence and published service/child recipes. A load recipe
specifies the required weight and context layout; `subagent_catalog` describes
which attached context can serve an agent. Gateway maps a main or agent
conversation onto a compatible recipe. A future recipe can place the main
context on another GPU, use separate model weights for agents, or support
multiple rigs without changing the conversational definitions above.
