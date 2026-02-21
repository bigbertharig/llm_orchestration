# CPU Agent Setup (Pi)

## Base Requirements
- `python3` installed
- Shared path mounted (`/media/bryan/shared`)
- Repo present at `/home/bryan/llm_orchestration`
- Optional but recommended: `~/ml-env` virtualenv

The agent automatically prepends:
- `source ~/ml-env/bin/activate && ...`
for CPU shell commands when `~/ml-env/bin/activate` exists.

## One-shot Test
```bash
python3 /media/bryan/shared/scripts/cpu_agent.py --config /media/bryan/shared/agents/config.json --once --name cpu-pi-test
```

## Continuous Run
```bash
python3 /media/bryan/shared/scripts/cpu_agent.py --config /media/bryan/shared/agents/config.json --name cpu-$(hostname)
```

## Heartbeat
CPU agent heartbeat is written to:
```bash
/media/bryan/shared/cpus/<agent_name>/heartbeat.json
```

## Systemd (recommended for Pi image)
1. Copy template:
```bash
sudo cp /home/bryan/llm_orchestration/scripts/cpu_agent.service.example /etc/systemd/system/cpu-agent.service
```
2. Enable/start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now cpu-agent.service
```
3. Check:
```bash
systemctl status cpu-agent.service --no-pager
tail -n 100 /media/bryan/shared/logs/cpu_workers/cpu-agent-$(hostname).log
```
