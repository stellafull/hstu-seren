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
MODEL_DIR=/root/autodl-fs/hf-models/Qwen-Qwen3-Embedding-8B
mkdir -p logs
run_embed() {
  local cfg="$1"
  echo "[SID] embedding ${cfg}"
  python src/generative_recommenders_pl/scripts/semantic_id_embed.py \
    embed_cfg="${cfg}" \
    +embedding.model.name="${MODEL_DIR}" \
    +embedding.model.device=cuda \
    +embedding.model.use_flash_attention=false
}
run_train() {
  local cfg="$1"
  echo "[SID] train ${cfg}"
  python src/generative_recommenders_pl/scripts/semantic_id_train.py \
    train_cfg="${cfg}" \
    +training.quantizer.device=cuda
}
run_infer() {
  local cfg="$1"
  echo "[SID] infer ${cfg}"
  python src/generative_recommenders_pl/scripts/semantic_id_infer.py \
    infer_cfg="${cfg}" \
    +inference.device=cuda \
    +inference.dedup_slot_cardinality=4096
}
run_embed embedding/emb_serendipity_2018
run_embed embedding/emb_amazon_movies
run_embed embedding/emb_amazon_books
run_train training/sid_train_serendipity_2018
run_train training/sid_train_amazon_movies
run_train training/sid_train_amazon_books
run_infer inference/sid_infer_serendipity_2018
run_infer inference/sid_infer_amazon_movies
run_infer inference/sid_infer_amazon_books
python src/generative_recommenders_pl/scripts/prepare_data.py data=serendipity_2018_training
python src/generative_recommenders_pl/scripts/prepare_data.py data=serendipity_2018_answers
python src/generative_recommenders_pl/scripts/prepare_data.py data=amazon_books
python src/generative_recommenders_pl/scripts/prepare_data.py data=serenlens_books
python src/generative_recommenders_pl/scripts/prepare_data.py data=amazon_movies
python src/generative_recommenders_pl/scripts/prepare_data.py data=serenlens_movies
python EDA/run_local_gate.py --all --skip-prepare
