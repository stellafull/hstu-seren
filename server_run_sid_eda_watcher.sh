#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/hstu-seren
LOG=/root/autodl-tmp/hstu-seren/server_finalize_gpu.log
PID=4007
while true; do
  if grep -q 'GPU_STACK_READY' "$LOG" 2>/dev/null && grep -q 'CUDA_SMOKE_OK' "$LOG" 2>/dev/null; then
    nohup bash /root/autodl-tmp/hstu-seren/server_run_sid_eda.sh > /root/autodl-tmp/hstu-seren/server_run_sid_eda.log 2>&1 &
    echo "SID_EDA_PID:$!"
    exit 0
  fi
  if ! ps -p "$PID" >/dev/null 2>&1; then
    echo 'GPU stack setup ended before readiness marker; not starting SID/EDA.'
    exit 1
  fi
  sleep 20
done
