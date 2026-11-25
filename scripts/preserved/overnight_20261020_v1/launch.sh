#!/bin/bash
set -euo pipefail
B=./logs/overnight_preflight_20261020_v1
P=./.pixi/envs/default/bin/python
[[ "${1:-}" == --approved ]] || { echo 'Requires explicit full-training approval'; exit 2; }
[[ ! -e "$B/BATCH_STARTED.json" ]] || { echo 'Already launched'; exit 2; }
exec tmux new-session -d -s vibration_overnight_20261020 "cd . && exec env CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 $P -B $B/batch.py --approved >$B/batch_console.log 2>&1"
