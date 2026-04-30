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
python -m pip install -U pip setuptools wheel
python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision
python -m pip install lightning torchmetrics hydra-core hydra-colorlog rich numpy pandas scipy scikit-learn pillow pyarrow tqdm omegaconf sentence-transformers transformers huggingface_hub hf_transfer
python - <<'PY'
mods=['torch','torchvision','lightning','torchmetrics','hydra','omegaconf','pandas','numpy','scipy','sklearn','PIL','pyarrow','sentence_transformers','transformers','huggingface_hub']
for m in mods:
    __import__(m)
print('IMPORT_CHECK_OK')
PY
