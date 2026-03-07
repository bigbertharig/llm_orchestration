# Ollama Pre-Update Snapshot

Captured: 2026-03-05 17:48 PST
Host: `bryan-GPU-Rig`

## Version

- `ollama version is 0.15.6`

## Binary

- command path: `/usr/local/bin/ollama`
- resolved path: `/usr/local/bin/ollama`

## Service Unit

Source:
- `/etc/systemd/system/ollama.service`

Observed `systemctl cat ollama`:

```ini
[Unit]
Description=Ollama Service
After=network-online.target

[Service]
ExecStart=/usr/local/bin/ollama serve
User=ollama
Group=ollama
Restart=always
RestartSec=3
Environment="PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin"

[Install]
WantedBy=default.target
```

## systemd Runtime Fields

- `FragmentPath=/etc/systemd/system/ollama.service`
- `DropInPaths=` (none)
- `ExecStart=/usr/local/bin/ollama serve`
- `Environment=PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin`

## Model Store Paths

- user store: `/home/bryan/.ollama`
- system store: `/usr/share/ollama/.ollama`

## Worker Runtime Inventory Before Update

Observed on worker endpoints `11435`-`11439`:

- `qwen3.5:4b`
- `qwen3.5:9b`
- `mistral:7b-instruct`
- `deepseek-r1:7b`
- `qwen2.5-coder:7b`
- `qwen2.5-coder:14b`
- `qwen2.5:7b`

## Worker State Before Update

After targeted resets, these workers were cold:

- `gpu-1`
- `gpu-2`
- `gpu-3`

This snapshot was taken after unloading/dropping the previously hot worker models in preparation for the Ollama update.

## Notes

- Dashboard `reset_gpu` was confirmed as the operator-facing hard reset path.
- Internal `reset_gpu_runtime` is a separate thermal-recovery path and was not used for the update workflow.
- `qwen3.5:4b` and `qwen3.5:9b` both failed direct `/api/generate` load on the current Ollama version before this update.
