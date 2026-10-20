# Run in an elevated PowerShell (Run as administrator).
# Undo setup_desktop_rig_ssh.ps1.
#   -Level pause      stop sshd and block inbound (easy to turn back on)
#   -Level remove     pause + delete the firewall rules, the Pi key and the WSL shell setting
#   -Level uninstall  remove + uninstall the OpenSSH server entirely
param([ValidateSet('pause', 'remove', 'uninstall')][string]$Level = 'pause')
$ErrorActionPreference = 'Stop'

# pause
if (Get-Service sshd -ErrorAction SilentlyContinue) {
    Stop-Service sshd -ErrorAction SilentlyContinue
    Set-Service sshd -StartupType Disabled
}
Get-NetFirewallRule -Name 'RigNet-SSH-In', 'RigNet-Ping-In' -ErrorAction SilentlyContinue | Disable-NetFirewallRule
if ($Level -eq 'pause') { Write-Host 'Paused: sshd stopped and disabled, rig-network rules disabled.'; return }

# remove
Get-NetFirewallRule -Name 'RigNet-SSH-In', 'RigNet-Ping-In' -ErrorAction SilentlyContinue | Remove-NetFirewallRule
$keys = 'C:\ProgramData\ssh\administrators_authorized_keys'
if (Test-Path $keys) { Remove-Item $keys -Force }
Remove-ItemProperty -Path 'HKLM:\SOFTWARE\OpenSSH' -Name DefaultShell -ErrorAction SilentlyContinue
if ($Level -eq 'remove') { Write-Host 'Removed: rules, authorized keys and WSL shell setting deleted; sshd disabled.'; return }

# uninstall
Remove-WindowsCapability -Online -Name 'OpenSSH.Server~~~~0.0.1.0' | Out-Null
Get-NetFirewallRule -Name 'OpenSSH-Server-In-TCP' -ErrorAction SilentlyContinue | Remove-NetFirewallRule
if (Test-Path 'C:\ProgramData\ssh') { Remove-Item 'C:\ProgramData\ssh' -Recurse -Force }
Write-Host 'Uninstalled: OpenSSH server removed with its config, keys and firewall rule.'
