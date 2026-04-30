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
export HF_HUB_ENABLE_HF_TRANSFER=1
python -m pip install -U pip setuptools wheel
python -m pip install huggingface_hub hf_transfer tqdm numpy pandas scipy scikit-learn pillow pyarrow omegaconf hydra-core hydra-colorlog rich safetensors transformers
python - <<'PY'
mods=['huggingface_hub','hf_transfer','tqdm','numpy','pandas','scipy','sklearn','PIL','pyarrow','omegaconf','hydra','rich','transformers','safetensors']
for m in mods:
    __import__(m)
print('BASE_IMPORT_CHECK_OK')
PY
python - <<'PY'
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id='Qwen/Qwen3-Embedding-8B',
    local_dir='/root/autodl-fs/hf-models/Qwen-Qwen3-Embedding-8B',
    resume_download=True,
)
print('MODEL_READY', path)
PY
