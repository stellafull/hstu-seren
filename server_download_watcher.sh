#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/hstu-seren
BOOTSTRAP_LOG=/root/autodl-tmp/hstu-seren/server_bootstrap.log
BOOTSTRAP_PID=1530
while true; do
  if grep -q 'IMPORT_CHECK_OK' "$BOOTSTRAP_LOG" 2>/dev/null; then
    bash /root/autodl-tmp/hstu-seren/server_predownload_qwen.sh
    exit 0
  fi
  if ! ps -p "$BOOTSTRAP_PID" >/dev/null 2>&1; then
    echo 'Bootstrap finished without IMPORT_CHECK_OK; aborting predownload.'
    exit 1
  fi
  sleep 20
done
