#!/usr/bin/env bash
# scripts/install.sh -- one-shot installer for Linux/macOS.
# Usage:  bash scripts/install.sh [121|124]   (default cu121)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

CUDA="${1:-121}"
echo "[install] cwd = $PROJECT_ROOT"
echo "[install] CUDA target = cu${CUDA}"

python -m pip install --upgrade pip
python -m pip install --index-url "https://download.pytorch.org/whl/cu${CUDA}" \
    torch torchvision torchaudio
python -m pip install -r requirements.txt

LOCKFILE="requirements.lock.txt"
echo "[install] freezing -> $LOCKFILE"
python -m pip freeze > "$LOCKFILE"

echo "[install] sanity check"
python - <<'PY'
import torch, importlib
print('torch        :', torch.__version__, '| cuda =', torch.cuda.is_available())
for mod in ['transformers', 'peft', 'bitsandbytes', 'datasets', 'safetensors',
            'accelerate', 'huggingface_hub']:
    m = importlib.import_module(mod)
    print(f'{mod:<14}: {getattr(m, "__version__", "?")}')
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f'GPU          : {p.name} | VRAM {p.total_memory/1e9:.1f} GB')
PY

echo "[install] done."
