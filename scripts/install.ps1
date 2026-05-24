# scripts/install.ps1 -- one-shot installer for the project's Python deps.
# Installs CUDA-enabled torch first (so bitsandbytes binds to the correct CUDA),
# then everything from requirements.txt. Run once after creating your venv.
#
# Usage:   pwsh -File scripts/install.ps1   [-Cuda 121 | 124]

param(
    [string]$Cuda = "121"
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$projectRoot = Split-Path -Parent $scriptDir
Set-Location $projectRoot

Write-Host "[install] cwd = $projectRoot" -ForegroundColor DarkGray
Write-Host "[install] CUDA target = cu$Cuda"

# 1) torch with CUDA wheels (must come before bitsandbytes)
Write-Host "[install] torch (CUDA $Cuda)" -ForegroundColor Cyan
python -m pip install --upgrade pip
python -m pip install --index-url "https://download.pytorch.org/whl/cu$Cuda" `
    torch torchvision torchaudio
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# 2) Everything else from requirements.txt
Write-Host "[install] remaining requirements" -ForegroundColor Cyan
python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# 3) Freeze a lockfile next to requirements.txt for reproducibility
$lockfile = "requirements.lock.txt"
Write-Host "[install] freezing -> $lockfile" -ForegroundColor Cyan
python -m pip freeze | Out-File -Encoding utf8 $lockfile

# 4) Quick sanity check
Write-Host "[install] verifying CUDA + bitsandbytes + peft..." -ForegroundColor Cyan
python -c @"
import torch, importlib
print('torch        :', torch.__version__, '| cuda =', torch.cuda.is_available())
for mod in ['transformers', 'peft', 'bitsandbytes', 'datasets', 'safetensors',
            'accelerate', 'huggingface_hub']:
    m = importlib.import_module(mod)
    v = getattr(m, '__version__', '?')
    print(f'{mod:<14}: {v}')
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f'GPU          : {p.name} | VRAM {p.total_memory/1e9:.1f} GB')
"@

Write-Host ""
Write-Host "[install] done." -ForegroundColor Green
