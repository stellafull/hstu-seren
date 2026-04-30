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
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=.:src
python src/generative_recommenders_pl/scripts/prepare_data.py data=serendipity_2018_answers
python src/generative_recommenders_pl/scripts/prepare_data.py data=amazon_books
python src/generative_recommenders_pl/scripts/prepare_data.py data=serenlens_books
python src/generative_recommenders_pl/scripts/prepare_data.py data=amazon_movies
python src/generative_recommenders_pl/scripts/prepare_data.py data=serenlens_movies
python EDA/run_local_gate.py --all --skip-prepare
