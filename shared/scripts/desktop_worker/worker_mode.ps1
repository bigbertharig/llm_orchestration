# Desktop worker mode switch for the Pi controller. Nothing here auto-starts.
#   worker_mode.ps1 on       keep WSL Ubuntu running (Docker starts with it) until 'off' or a Windows restart
#   worker_mode.ps1 off      stop model containers, end the keepalive, shut Ubuntu down
#   worker_mode.ps1 status   keepalive, Docker and containers
param([ValidateSet('on', 'off', 'status')][string]$Action = 'status')
$Distro = 'Ubuntu'
$Tag = 'desktop-rig-keepalive'

function Get-Keepalive {
    Get-CimInstance Win32_Process -Filter "Name = 'wsl.exe'" |
        Where-Object { $_.CommandLine -like "*$Tag*" }
}

function Show-Status {
    $k = @(Get-Keepalive)
    Write-Host ('keepalive: ' + $(if ($k.Count) { "running (pid $($k.ProcessId -join ','))" } else { 'off' }))
    $running = (wsl.exe -l --running 2>$null | Out-String) -replace "`0", ''
    if ($running -notmatch $Distro) { Write-Host "ubuntu: stopped"; return }
    # No double quotes in commands passed to wsl.exe: Windows PowerShell 5.1 mangles them
    wsl.exe -d $Distro -u root -e bash -c 'echo docker: $(systemctl is-active docker); docker ps --format ''container: {{.Names}} {{.Status}}'' 2>/dev/null' 2>$null |
        Where-Object { $_ -notmatch 'systemd user session' }
}

switch ($Action) {
    'on' {
        if (-not (Get-Keepalive)) {
            # One string: Windows PowerShell joins an argument array without quoting it
            Start-Process -WindowStyle Hidden -FilePath 'C:\Windows\System32\wsl.exe' `
                -ArgumentList "-d $Distro -e bash -c `"exec -a $Tag sleep infinity`""
        }
        Write-Host 'waiting for Docker...'
        $deadline = (Get-Date).AddSeconds(180)
        do {
            Start-Sleep 3
            $ok = wsl.exe -d $Distro -u root -e bash -c 'docker info >/dev/null 2>&1 && echo up' 2>$null
        } until ($ok -match 'up' -or (Get-Date) -gt $deadline)
        if ($ok -notmatch 'up') { Write-Host 'Docker did not come up within 180 s'; Show-Status; exit 1 }
        Write-Host 'worker mode ON'
        Show-Status
    }
    'off' {
        $running = (wsl.exe -l --running 2>$null | Out-String) -replace "`0", ''
        if ($running -match $Distro) {
            wsl.exe -d $Distro -u root -e bash -c 'docker ps -q | xargs -r docker stop >/dev/null; echo containers stopped' 2>$null |
                Where-Object { $_ -notmatch 'systemd user session' }
        }
        Get-Keepalive | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
        wsl.exe --terminate $Distro | Out-Null
        Write-Host 'worker mode OFF'
    }
    'status' { Show-Status }
}
