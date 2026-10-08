# PoRace trainer launcher. Usage (from repo root):
#   powershell -File training\run.ps1 -Gpu train.py --env joystick --num_envs 4096 --logdir runs/r1 --restore runs/r1/checkpoints
#   powershell -File training\run.ps1 eval.py --rung r1 --params runs/r1/params.pkl
# --init          : reaps child processes so a hung run can be stopped (a zombie cost a Docker restart on 2026-10-07)
# --log-driver none: Warp prints a solver warning per step; do not let it fill the container log
# --name          : fixed names so a run can be found and stopped
param([switch]$Gpu, [string]$Name = "", [Parameter(ValueFromRemainingArguments = $true)][string[]]$Rest)
$root = Split-Path -Parent $PSScriptRoot
$a = @("run", "--rm", "--init", "--log-driver", "none",
       "-v", "$root\training:/work", "-v", "$root\docs:/docs", "-v", "$root\Assets:/assets")
if ($Name) { $a += @("--name", $Name) }
if ($Gpu) { $a += @("--gpus", "all") } else { $a += @("-e", "JAX_PLATFORMS=cpu") }
$a += @("porace-trainer", "python") + $Rest
& docker @a 2>&1 | Where-Object { $_ -notmatch 'iterations limit reached|To disable the print|^\s*$|CUDA|NVIDIA|container image|nvidia.com|license|By pulling|convenience' }
exit $LASTEXITCODE
