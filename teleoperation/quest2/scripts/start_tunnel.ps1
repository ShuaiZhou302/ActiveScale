<#
.SYNOPSIS
  Keeps SSH port forwards to the robot host alive for Quest teleop.

.DESCRIPTION
  Reuses an SSH config host entry and forwards two ports
  over one SSH connection:
    - Port        (default 8770): pose/button stream, webxr_server.py ->
                  quest_server.py (127.0.0.1 only on the robot side).
    - CameraPort  (default 8771): camera stream, the OTHER direction --
                  camera_streamer.py -> webxr_server.py (also 127.0.0.1
                  only on the robot side). Deliberately a separate port
                  from Port, see protocol.py's module docstring for why.

  Windows has no autossh, so this is a plain retry loop: if ssh exits for
  any reason (VPN blip, sleep/wake, robot reboot), it reconnects after a
  short delay. Run this in its own terminal and leave it open for the whole
  teleop session; webxr_server.py does its own reconnect on top of this.

.PARAMETER Port
  Local AND remote port for the pose forward (must match quest_server.py --port).

.PARAMETER CameraPort
  Local AND remote port for the camera forward (must match camera_streamer.py
  --port and webxr_server.py --camera_port).

.PARAMETER SshHost
  SSH config alias to use. Defaults to 'robot'.

.EXAMPLE
  powershell -File scripts\start_tunnel.ps1
  powershell -File scripts\start_tunnel.ps1 -Port 8770 -CameraPort 8771 -SshHost robot
#>
param(
    [int]$Port = 8770,
    [int]$CameraPort = 8771,
    [string]$SshHost = "robot"
)

Write-Output "Quest teleop tunnel: 127.0.0.1:{$Port,$CameraPort} <-> ${SshHost}:{$Port,$CameraPort}"
Write-Output "Ctrl-C to stop. Reconnects automatically on drop."

while ($true) {
    $start = Get-Date
    Write-Output "[$start] connecting..."

    # -N: no remote command, just forward. -L: local forward (repeat per port).
    # ExitOnForwardFailure / ServerAliveInterval already set on the host in
    # ~/.ssh/config, so a dead link is detected and this process exits,
    # letting the loop below reconnect.
    ssh -N `
        -L 127.0.0.1:${Port}:127.0.0.1:${Port} `
        -L 127.0.0.1:${CameraPort}:127.0.0.1:${CameraPort} `
        $SshHost

    $elapsed = (Get-Date) - $start
    Write-Output "[$(Get-Date)] tunnel dropped after $($elapsed.TotalSeconds.ToString('F0'))s, reconnecting in 2s..."
    Start-Sleep -Seconds 2
}
