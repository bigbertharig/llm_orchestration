# Desktop worker (Windows 10 + WSL2)

Turns a Windows desktop with an NVIDIA card into a second rig the Pi can reach over ssh. First set up
2026-10-19 on the GTX 1070 desktop (10.0.0.4). Background and test results: llm-gateway
`docs/DESIGN_NOTES.md`, "More than one rig"; host notes: `shared/workspace/systems_prep.md` item 9.

| File | Runs on | Does |
|---|---|---|
| `setup_desktop_rig_ssh.ps1` | Windows, elevated | OpenSSH server, `bash.exe` default shell (lands in WSL), key-only, firewall open to 10.0.0.0/24 on the Ethernet adapter only |
| `teardown_desktop_rig_ssh.ps1` | Windows, elevated | `-Level pause` / `remove` / `uninstall` (removes the OpenSSH server) |
| `wsl_docker_setup.sh` | WSL, as root | native `docker-ce` + `nvidia-container-toolkit`, GPU check |
| `worker_mode.ps1` | Windows | `on` / `off` / `status`; `worker_on.cmd`, `worker_off.cmd` for double-click |

## Setup

1. Windows NVIDIA driver on the 580 branch (582.78 for Pascal). WSL uses the Windows driver, and
   `llama-runtime:b11669` needs CUDA 12.9. Check: `nvidia-smi` shows "CUDA Version: 13.0".
2. Copy the controller's public key next to the setup script as `pi_id_ed25519.pub` (gitignored),
   then in an elevated PowerShell: `powershell -ExecutionPolicy Bypass -File setup_desktop_rig_ssh.ps1`.
3. In WSL Ubuntu with `systemd=true` in `/etc/wsl.conf`:
   `wsl.exe -d Ubuntu -u root -e bash wsl_docker_setup.sh`.
4. Runtime image from the rig: `ssh rig 'docker save llama-runtime:b11669' | wsl.exe -d Ubuntu -u root -e docker load`.
5. Models on the WSL disk (for example `~/models/<family>/`); reading over NFS on gigabit is about 110 MB/s.
6. On the Pi: a `desktop-rig` entry in `~/.ssh/config` (HostName 10.0.0.4, User bryan). Check with
   `ssh desktop-rig nvidia-smi -L`.

## Worker mode

WSL stops the Ubuntu instance when no Windows process holds it, and that stops Docker and every
container. `worker_mode.ps1 on` starts a hidden keepalive and waits for Docker (about 40 s from a cold
start). It stays on until `off` or a Windows restart. Nothing starts at boot on purpose: the desktop is
a worker only when switched on. ssh still works with worker mode off, but lands on a cold instance.

## Windows quirks

- Windows 10 has no WSL mirrored networking, so sshd inside WSL is not reachable from the LAN; the
  Windows OpenSSH server with `bash.exe` as the shell is the way in.
- Model ports reach the Pi with `ssh -L <pi-port>:127.0.0.1:<port> desktop-rig` (WSL localhost
  forwarding) while worker mode is on.
- Windows PowerShell 5.1 mangles double quotes in arguments passed to `wsl.exe`, and joins a
  `Start-Process -ArgumentList` array without quoting; the scripts avoid both.
- A harmless "Failed to start the systemd user session" line appears on ssh logins.

## Turning it off

`worker_off.cmd`, then `teardown_desktop_rig_ssh.ps1 -Level pause|remove|uninstall` for ssh.
