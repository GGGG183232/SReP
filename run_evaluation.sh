#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export KEEA_GPUS="${KEEA_GPUS:-0,1,2,3}"
export KEEA_NUM_WORKERS="${KEEA_NUM_WORKERS:-20}"
export KEEA_TIMEOUT="${KEEA_TIMEOUT:-600}"
export KEEA_MAX_BUDGET="${KEEA_MAX_BUDGET:-500}"
export KEEA_SEED="${KEEA_SEED:-20260920}"
export KEEA_DATASETS="${KEEA_DATASETS:-USPTO,Ames,BBBP,ClinTox,Toxicity_LD50,logP,logS}"
export KEEA_CPU_THREADS="${KEEA_CPU_THREADS:-1}"
export KEEA_DEPTH_PENALTY_START="${KEEA_DEPTH_PENALTY_START:-20}"
export KEEA_DEPTH_PENALTY_STRONG_START="${KEEA_DEPTH_PENALTY_STRONG_START:-40}"
export KEEA_DEPTH_PENALTY_SOFT_SLOPE="${KEEA_DEPTH_PENALTY_SOFT_SLOPE:-0.05}"
export KEEA_DEPTH_PENALTY_STRONG_SLOPE="${KEEA_DEPTH_PENALTY_STRONG_SLOPE:-1.0}"
export OMP_NUM_THREADS="$KEEA_CPU_THREADS"
export MKL_NUM_THREADS="$KEEA_CPU_THREADS"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

KEEA_PYTHON="${KEEA_PYTHON:-python}"
"$KEEA_PYTHON" -u RetrosynthesisPlanning/evaluate_structure.py --preflight
if [[ "${KEEA_PREFLIGHT_ONLY:-0}" == 1 ]]; then
    exit 0
fi
mkdir -p -- runs
KEEA_RUN_DIR=$(mktemp -d "$PWD/runs/srep_b${KEEA_MAX_BUDGET}_$(date +%Y%m%d_%H%M%S)_XXXXXX")
KEEA_LOG_FILE="$KEEA_RUN_DIR/evaluation.log"
nohup "$KEEA_PYTHON" -u RetrosynthesisPlanning/evaluate_structure.py \
    --run-dir "$KEEA_RUN_DIR" > "$KEEA_LOG_FILE" 2>&1 < /dev/null &
KEEA_PID=$!
printf 'PID: %s\nResults: %s/test\nTraces: %s/traces\nConfig: %s/run_config.json\nLog: %s\n' \
    "$KEEA_PID" "$KEEA_RUN_DIR" "$KEEA_RUN_DIR" "$KEEA_RUN_DIR" "$KEEA_LOG_FILE"
printf "Progress: tail --pid=%s -n +1 -f '%s'\n" "$KEEA_PID" "$KEEA_LOG_FILE"
