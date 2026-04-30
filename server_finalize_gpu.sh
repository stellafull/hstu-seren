#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/hstu-seren
source .venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-fs/hf-home
export HUGGINGFACE_HUB_CACHE=/root/autodl-fs/hf-home/hub
export HF_HUB_CACHE=/root/autodl-fs/hf-home/hub
export TRANSFORMERS_CACHE=/root/autodl-fs/hf-home/transformers
export PIP_CACHE_DIR=/root/autodl-fs/pip-cache
python -m pip uninstall -y torch torchvision torchaudio || true
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
python -m pip install lightning torchmetrics sentence-transformers
python - <<'PY'
import torch, torchvision, lightning, torchmetrics, sentence_transformers
print('GPU_STACK_READY', torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO_GPU')
x = torch.randn(4, 4, device='cuda')
y = torch.randn(4, 4, device='cuda')
print('CUDA_SMOKE_OK', (x @ y)[0,0].item())
PY
