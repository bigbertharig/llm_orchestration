# Run in an elevated PowerShell (Run as administrator).
# Lets the Pi ssh into this desktop and land in WSL Ubuntu. Key-only, rig network only.
$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path

# 1. Install and start the Windows OpenSSH server
if ((Get-WindowsCapability -Online -Name 'OpenSSH.Server*').State -ne 'Installed') {
    Add-WindowsCapability -Online -Name 'OpenSSH.Server~~~~0.0.1.0' | Out-Null
}
Set-Service sshd -StartupType Automatic
Start-Service sshd

# 2. Sessions open in WSL (default distro: Ubuntu)
New-ItemProperty -Path 'HKLM:\SOFTWARE\OpenSSH' -Name DefaultShell -Value 'C:\Windows\System32\bash.exe' -PropertyType String -Force | Out-Null

# 3. Key-only login; only the Pi's key is authorized (admin accounts use this file)
$cfg = 'C:\ProgramData\ssh\sshd_config'
$text = Get-Content $cfg -Raw
$text = $text -replace '(?m)^#?\s*PasswordAuthentication\s+\S+', 'PasswordAuthentication no'
if ($text -notmatch '(?m)^PasswordAuthentication no') { $text = "PasswordAuthentication no`r`n" + $text }
$text = $text -replace '(?m)^#?\s*PubkeyAuthentication\s+\S+', 'PubkeyAuthentication yes'
$text = $text -replace '(?m)^#?\s*(KbdInteractiveAuthentication|ChallengeResponseAuthentication)\s+\S+\r?\n?', ''
$text = "KbdInteractiveAuthentication no`r`n" + $text
Set-Content -Path $cfg -Value $text -Encoding ascii

$keys = 'C:\ProgramData\ssh\administrators_authorized_keys'
$piKey = (Get-Content (Join-Path $here 'pi_id_ed25519.pub') -Raw).Trim()
$existing = if (Test-Path $keys) { Get-Content $keys } else { @() }
if ($existing -notcontains $piKey) { Add-Content -Path $keys -Value $piKey -Encoding ascii }
icacls $keys /inheritance:r /grant 'Administrators:F' /grant 'SYSTEM:F' | Out-Null

# 4. Firewall: ssh and ping from the rig network only
Get-NetFirewallRule -Name 'OpenSSH-Server-In-TCP' -ErrorAction SilentlyContinue | Disable-NetFirewallRule
# Wired adapter only: nothing over Wi-Fi or VPN, even from a 10.0.0.x address
Get-NetFirewallRule -Name 'RigNet-SSH-In', 'RigNet-Ping-In' -ErrorAction SilentlyContinue | Remove-NetFirewallRule
New-NetFirewallRule -Name 'RigNet-SSH-In' -DisplayName 'SSH from rig network (Ethernet, 10.0.0.0/24)' `
    -Direction Inbound -Protocol TCP -LocalPort 22 -RemoteAddress 10.0.0.0/24 -InterfaceAlias 'Ethernet' `
    -Profile Private -Action Allow | Out-Null
New-NetFirewallRule -Name 'RigNet-Ping-In' -DisplayName 'Ping from rig network (Ethernet, 10.0.0.0/24)' `
    -Direction Inbound -Protocol ICMPv4 -IcmpType 8 -RemoteAddress 10.0.0.0/24 -InterfaceAlias 'Ethernet' `
    -Profile Private -Action Allow | Out-Null

Restart-Service sshd
Write-Host 'Done. sshd:' (Get-Service sshd).Status
