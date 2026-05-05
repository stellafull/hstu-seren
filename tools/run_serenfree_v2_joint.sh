#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/root/autodl-tmp/hstu-seren}"
RUN_ID="${RUN_ID:-$(date -u '+%Y%m%d_%H%M%S')}"
RUN_DIR="$ROOT/logs/serenfree_v2_rai_${RUN_ID}"
CKPT_ROOT="$ROOT/tmp/checkpoints/serenfree_v2_rai_${RUN_ID}"
PY="$ROOT/.venv/bin/python"
KS=(10 20 50 100 200)
ALPHAS=(${ALPHA_GRID:-0.0 0.02 0.05 0.1 0.2})

if [[ -n "${TRAIN_BATCH_CANDIDATES:-}" ]]; then
  read -r -a TRAIN_BATCHES <<< "$TRAIN_BATCH_CANDIDATES"
else
  TRAIN_BATCHES=(1024 768 512 256)
fi
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
EVAL_BEAM_SIZE="${EVAL_BEAM_SIZE:-200}"
EVAL_CANDIDATE_M="${EVAL_CANDIDATE_M:-200}"
EVAL_RELEVANCE_FLOOR_RANK="${EVAL_RELEVANCE_FLOOR_RANK:-200}"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-20}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-true}"
TRAIN_VAL_BEAM_SIZE="${TRAIN_VAL_BEAM_SIZE:-100}"
TRAIN_VAL_EVAL_BATCH_SIZE="${TRAIN_VAL_EVAL_BATCH_SIZE:-512}"
TRAIN_VAL_KS="${TRAIN_VAL_KS:-[10,20,50,100]}"
CHECK_VAL_EVERY_N_EPOCH="${CHECK_VAL_EVERY_N_EPOCH:-5}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-3}"
TRAIN_MAX_SEQUENCE_LENGTH="${TRAIN_MAX_SEQUENCE_LENGTH:-100}"
MAX_EPOCHS_R="${MAX_EPOCHS_R:-100}"
MAX_EPOCHS_RAI="${MAX_EPOCHS_RAI:-100}"

cd "$ROOT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
export TORCHINDUCTOR_CACHE_DIR="$ROOT/tmp/torchinductor_cache"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.8}"

mkdir -p "$RUN_DIR" "$CKPT_ROOT" "$ROOT/tmp/loo_manifest" "$ROOT/tmp/ser_event_manifest" "$ROOT/tmp/torchinductor_cache"

log() {
  printf '[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" | tee -a "$RUN_DIR/commands.log"
}

run_logged() {
  local name="$1"
  shift
  local logfile="$RUN_DIR/${name}.log"
  log "RUN $name"
  printf '[%s] CMD:' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >> "$logfile"
  printf ' %q' "$@" >> "$logfile"
  printf '\n' >> "$logfile"
  "$@" >> "$logfile" 2>&1
}

run_tty_logged() {
  local name="$1"
  shift
  local logfile="$RUN_DIR/${name}.log"
  log "RUN $name"
  printf '[%s] CMD:' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >> "$logfile"
  printf ' %q' "$@" >> "$logfile"
  printf '\n' >> "$logfile"
  script -q -e -f -a "$logfile" -c "$(printf '%q ' "$@")"
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    log "MISSING $path"
    return 1
  fi
}

build_loo() {
  local label="$1"
  local input="$2"
  local sid="$3"
  local out="$4"
  require_file "$input"
  require_file "$sid"
  run_logged "build_loo_${label}" \
    "$PY" tools/build_loo_manifest.py \
    --dataset "$label" \
    --input "$input" \
    --output-dir "$out" \
    --split-version loo_v1 \
    --sid-lookup "$sid" \
    --sid-item-shift 1
}

build_ser_event() {
  local label="$1"
  local input="$2"
  local sid="$3"
  local out="$4"
  require_file "$input"
  require_file "$sid"
  run_logged "build_ser_event_${label}" \
    "$PY" tools/build_ser_event_manifest.py \
    --dataset "$label" \
    --input "$input" \
    --output-dir "$out" \
    --split-version ser_event_v1 \
    --sid-lookup "$sid" \
    --sid-item-shift 1 \
    --min-prefix 1 \
    --rating-threshold 4.0
}

gpu_preflight() {
  local label="$1"
  local experiment="$2"
  local manifest_dir="$3"
  run_logged "preflight_${label}_${experiment}" \
    "$PY" tools/serenfree_v2_gpu_preflight.py \
    --experiment "$experiment" \
    --override "data.train_dataset.ratings_file=${manifest_dir}/loo_train.parquet" \
    --override "data.val_dataset.ratings_file=${manifest_dir}/loo_eval.parquet" \
    --override "data.max_sequence_length=${TRAIN_MAX_SEQUENCE_LENGTH}" \
    --override data.batch_size=8 \
    --override data.num_workers=0
}

train_model() {
  local label="$1"
  local phase="$2"
  local experiment="$3"
  local manifest_dir="$4"
  local ckpt_dir="$5"
  local max_epochs="$6"
  local init_ckpt="${7:-}"
  local resume_ckpt="${8:-}"
  local status=0
  local batch_size
  for batch_size in "${TRAIN_BATCHES[@]}"; do
    log "TRY train_${label}_${phase} batch_size=${batch_size}"
    local cmd=(
      "$PY" src/generative_recommenders_pl/scripts/train.py
      "experiment=${experiment}"
      trainer=gpu
      test=false
      data.max_sequence_length="$TRAIN_MAX_SEQUENCE_LENGTH"
      data.batch_size="$batch_size"
      data.num_workers="$TRAIN_NUM_WORKERS"
      data.prefetch_factor="$PREFETCH_FACTOR"
      data.pin_memory=true
      +data.persistent_workers="$PERSISTENT_WORKERS"
      trainer.max_epochs="$max_epochs"
      trainer.check_val_every_n_epoch="$CHECK_VAL_EVERY_N_EPOCH"
      +trainer.num_sanity_val_steps=0
      model.validation_beam_size="$TRAIN_VAL_BEAM_SIZE"
      model.validation_eval_batch_size="$TRAIN_VAL_EVAL_BATCH_SIZE"
      "model.validation_ks=${TRAIN_VAL_KS}"
      callbacks.model_checkpoint.dirpath="$ckpt_dir"
      callbacks.model_checkpoint.monitor=val/ndcg@100
      callbacks.model_checkpoint.mode=max
      callbacks.early_stopping.monitor=val/ndcg@100
      callbacks.early_stopping.mode=max
      callbacks.early_stopping.patience="$EARLY_STOPPING_PATIENCE"
      "data.train_dataset.ratings_file=${manifest_dir}/loo_train.parquet"
      "data.val_dataset.ratings_file=${manifest_dir}/loo_eval.parquet"
      "+logger.tensorboard.version=${RUN_ID}_${label}_${phase}"
      "+logger.csv.version=${RUN_ID}_${label}_${phase}"
    )
    if [[ -n "$init_ckpt" ]]; then
      cmd+=("init_from_checkpoint=${init_ckpt}")
    fi
    if [[ -n "$resume_ckpt" ]]; then
      cmd+=("ckpt_path=${resume_ckpt}")
    fi
    if run_tty_logged "train_${label}_${phase}_bs${batch_size}" "${cmd[@]}"; then
      printf '%s\n' "$batch_size" > "$RUN_DIR/${label}_${phase}.batch_size"
      return 0
    else
      status=$?
    fi
    if grep -qiE 'out of memory|CUBLAS_STATUS_ALLOC_FAILED|CUDNN_STATUS_ALLOC_FAILED' "$RUN_DIR/train_${label}_${phase}_bs${batch_size}.log"; then
      log "OOM train_${label}_${phase} batch_size=${batch_size}; retry smaller batch"
      rm -rf "$ckpt_dir"
      continue
    fi
    return "$status"
  done
  log "FAILED train_${label}_${phase}: all batch candidates exhausted"
  return 1
}

best_checkpoint() {
  local ckpt_dir="$1"
  local best
  best=$(find "$ckpt_dir" -maxdepth 1 -type f -name 'epoch_*.ckpt' -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
  if [[ -n "$best" ]]; then
    printf '%s\n' "$best"
  else
    printf '%s\n' "$ckpt_dir/last.ckpt"
  fi
}

eval_manifest() {
  local label="$1"
  local phase="$2"
  local experiment="$3"
  local checkpoint="$4"
  local manifest_dir="$5"
  local parquet_name="$6"
  local alpha="$7"
  local output_json="$8"
  require_file "$checkpoint"
  require_file "$manifest_dir/$parquet_name"
  run_logged "eval_${label}_${phase}_alpha${alpha//./p}" \
    "$PY" src/generative_recommenders_pl/scripts/evaluate_serenfree_retrieval.py \
    --experiment "$experiment" \
    --checkpoint "$checkpoint" \
    --split test \
    --device cuda \
    --beam-size "$EVAL_BEAM_SIZE" \
    --ks "${KS[@]}" \
    --output-json "$output_json" \
    --late-fusion \
    --aig-alpha "$alpha" \
    --candidate-M "$EVAL_CANDIDATE_M" \
    --relevance-floor-rank "$EVAL_RELEVANCE_FLOOR_RANK" \
    --beta 0.0 \
    --progress-every 10 \
    --override "serenfree_eval.manifest_dir=${manifest_dir}" \
    --override "data.test_dataset.ratings_file=${manifest_dir}/${parquet_name}" \
    --override "data.max_sequence_length=${TRAIN_MAX_SEQUENCE_LENGTH}" \
    --override data.batch_size="$EVAL_BATCH_SIZE" \
    --override data.num_workers=8
}

eval_alpha_grid() {
  local label="$1"
  local experiment_loo="$2"
  local experiment_ser="$3"
  local checkpoint="$4"
  local normal_loo="$5"
  local tuning_loo="$6"
  local ser_event="$7"
  local alpha
  for alpha in "${ALPHAS[@]}"; do
    eval_manifest "$label" "normal_relevance" "$experiment_loo" "$checkpoint" "$normal_loo" "loo_eval.parquet" "$alpha" "$RUN_DIR/${label}_normal_relevance_alpha${alpha//./p}.json"
    eval_manifest "$label" "tuning_relevance" "$experiment_loo" "$checkpoint" "$tuning_loo" "loo_eval.parquet" "$alpha" "$RUN_DIR/${label}_tuning_relevance_alpha${alpha//./p}.json"
    eval_manifest "$label" "tuning_ser_event" "$experiment_ser" "$checkpoint" "$ser_event" "test_eval.parquet" "$alpha" "$RUN_DIR/${label}_tuning_ser_event_alpha${alpha//./p}.json"
  done
}

run_dataset() {
  local label="$1"
  local exp_r="$2"
  local exp_rai="$3"
  local exp_eval_loo="$4"
  local exp_eval_ser="$5"
  local train_input="$6"
  local tuning_input="$7"
  local sid="$8"

  local train_loo="$ROOT/tmp/loo_manifest/${label}/loo_v1"
  local tune_loo="$ROOT/tmp/loo_manifest/${label}_tuning/loo_v1"
  local ser_event="$ROOT/tmp/ser_event_manifest/${label}_tuning/ser_event_v1"
  local r_ckpt_dir="$CKPT_ROOT/${label}/r_only"
  local rai_ckpt_dir="$CKPT_ROOT/${label}/rai_policy"
  local resume_var="RESUME_${label^^}_R_CKPT"
  local resume_r_ckpt="${!resume_var-}"
  local existing_r_var="EXISTING_${label^^}_R_CKPT"
  local existing_r_ckpt="${!existing_r_var-}"
  local existing_rai_var="EXISTING_${label^^}_RAI_CKPT"
  local existing_rai_ckpt="${!existing_rai_var-}"

  log "DATASET ${label}: build normal LOO manifest"
  rm -rf "$train_loo"
  build_loo "$label" "$train_input" "$sid" "$train_loo"

  log "DATASET ${label}: build tuning relevance and ser-event manifests"
  rm -rf "$tune_loo" "$ser_event"
  build_loo "${label}_tuning" "$tuning_input" "$sid" "$tune_loo"
  build_ser_event "${label}_tuning" "$tuning_input" "$sid" "$ser_event"

  log "DATASET ${label}: GPU preflight"
  gpu_preflight "$label" "$exp_r" "$train_loo"
  gpu_preflight "$label" "$exp_rai" "$train_loo"

  local r_ckpt
  if [[ -n "$existing_r_ckpt" ]]; then
    require_file "$existing_r_ckpt"
    r_ckpt="$existing_r_ckpt"
    log "DATASET ${label}: use existing R checkpoint ${r_ckpt}"
  else
    log "DATASET ${label}: train R-only-new-loader"
    if [[ -n "$resume_r_ckpt" ]]; then
      require_file "$resume_r_ckpt"
      mkdir -p "$r_ckpt_dir"
      log "DATASET ${label}: resume R-only from ${resume_r_ckpt}"
    else
      rm -rf "$r_ckpt_dir"
    fi
    train_model "$label" "r_only" "$exp_r" "$train_loo" "$r_ckpt_dir" "$MAX_EPOCHS_R" "$resume_r_ckpt"
    r_ckpt="$(best_checkpoint "$r_ckpt_dir")"
  fi
  log "DATASET ${label}: R checkpoint ${r_ckpt}"

  local rai_ckpt
  if [[ -n "$existing_rai_ckpt" ]]; then
    require_file "$existing_rai_ckpt"
    rai_ckpt="$existing_rai_ckpt"
    log "DATASET ${label}: use existing RAI checkpoint ${rai_ckpt}"
  else
    log "DATASET ${label}: train RAI heads with aux detach from R checkpoint"
    rm -rf "$rai_ckpt_dir"
    train_model "$label" "rai_policy" "$exp_rai" "$train_loo" "$rai_ckpt_dir" "$MAX_EPOCHS_RAI" "$r_ckpt"
    rai_ckpt="$(best_checkpoint "$rai_ckpt_dir")"
  fi
  log "DATASET ${label}: RAI checkpoint ${rai_ckpt}"

  log "DATASET ${label}: alpha grid report"
  eval_alpha_grid "$label" "$exp_eval_loo" "$exp_eval_ser" "$rai_ckpt" "$train_loo" "$tune_loo" "$ser_event"
}

summarize_results() {
  run_logged summarize_results "$PY" - "$RUN_DIR" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
ks = [10, 20, 50, 100, 200]
for path in sorted(run_dir.glob("*.json")):
    data = json.loads(path.read_text())
    metrics = data.get("metrics", {})
    all_metrics = metrics.get("all", {})
    ser_metrics = metrics.get("ser_targets_only", {})
    counts = metrics.get("counts", {})
    row = {
        "file": path.name,
        "count_all": counts.get("all"),
        "count_ser": counts.get("ser_label"),
    }
    for k in ks:
        if f"hr@{k}" in all_metrics:
            row[f"HR@{k}"] = all_metrics[f"hr@{k}"]
            row[f"NDCG@{k}"] = all_metrics[f"ndcg@{k}"]
        if f"hr_ser@{k}" in ser_metrics:
            row[f"HRser@{k}"] = ser_metrics[f"hr_ser@{k}"]
            row[f"NDCGser@{k}"] = ser_metrics[f"ndcg_ser@{k}"]
    print(json.dumps(row, sort_keys=True))
PY
}

log "START SerenFree V2 R/A/I next-transition policy run id=$RUN_ID"
log "RUN_DIR=$RUN_DIR"
log "CKPT_ROOT=$CKPT_ROOT"
log "Policy: R-only full run, then RAI aux-detach heads, then alpha grid report. SER labels are report-only."
log "Train batch candidates=${TRAIN_BATCHES[*]} train_workers=${TRAIN_NUM_WORKERS} prefetch=${PREFETCH_FACTOR} persistent_workers=${PERSISTENT_WORKERS}"
log "Train max_sequence_length=${TRAIN_MAX_SEQUENCE_LENGTH}"
log "Train validation beam=${TRAIN_VAL_BEAM_SIZE} validation_eval_batch_size=${TRAIN_VAL_EVAL_BATCH_SIZE} validation_ks=${TRAIN_VAL_KS} check_val_every_n_epoch=${CHECK_VAL_EVERY_N_EPOCH} early_stopping_patience=${EARLY_STOPPING_PATIENCE}"
log "max_epochs_r=${MAX_EPOCHS_R} max_epochs_rai=${MAX_EPOCHS_RAI} eval_batch_size=${EVAL_BATCH_SIZE} eval_beam=${EVAL_BEAM_SIZE} eval_candidate_M=${EVAL_CANDIDATE_M} eval_relevance_floor=${EVAL_RELEVANCE_FLOOR_RANK} alpha_grid=${ALPHAS[*]}"

(
  nvidia-smi \
    --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
    --format=csv \
    -l 30 \
    > "$RUN_DIR/nvidia_smi.csv" 2>&1
) &
NVIDIA_SMI_PID=$!
trap 'kill "$NVIDIA_SMI_PID" 2>/dev/null || true' EXIT

run_dataset \
  movielens \
  serenfree_v2_r_pretrain_movielens \
  serenfree_v2_future_ai_movielens \
  serenfree_v2_eval_loo_movielens \
  serenfree_v2_eval_ser_event_movielens \
  "$ROOT/tmp/pre_train/serendipity-2018/sasrec_format.csv" \
  "$ROOT/tmp/tuning/serendipity-2018/sasrec_format.csv" \
  "$ROOT/tmp/semantic_id/serendipity_2018/item_sid_lookup_shifted.pt"

run_dataset \
  movies \
  serenfree_v2_r_pretrain_movies \
  serenfree_v2_future_ai_movies \
  serenfree_v2_eval_loo_movies \
  serenfree_v2_eval_ser_event_movies \
  "$ROOT/tmp/pre_train/amazon_movies/sasrec_format.csv" \
  "$ROOT/tmp/tuning/amazon_movies/sasrec_format.csv" \
  "$ROOT/tmp/semantic_id/amazon_movies/item_sid_lookup_shifted.pt"

run_dataset \
  books \
  serenfree_v2_r_pretrain_books \
  serenfree_v2_future_ai_books \
  serenfree_v2_eval_loo_books \
  serenfree_v2_eval_ser_event_books \
  "$ROOT/tmp/pre_train/amazon_books/sasrec_format.csv" \
  "$ROOT/tmp/tuning/amazon_books/sasrec_format.csv" \
  "$ROOT/tmp/semantic_id/amazon_books/item_sid_lookup_shifted.pt"

summarize_results
log "DONE SerenFree V2 R/A/I next-transition policy run"
