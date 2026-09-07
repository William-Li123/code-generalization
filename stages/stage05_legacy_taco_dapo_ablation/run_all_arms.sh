#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
: "${CG_DATA_ROOT:?set CG_DATA_ROOT to the external dataset root}"
: "${CG_MODEL_ROOT:?set CG_MODEL_ROOT to the external model root}"
: "${CG_OUTPUT_ROOT:?set CG_OUTPUT_ROOT to the external output root}"
CG_WORK_ROOT="${CG_WORK_ROOT:-$CG_OUTPUT_ROOT/.work}"
VERL_RECIPE_DIR="${VERL_RECIPE_DIR:-${CG_VERL_RECIPE_DIR:-}}"
: "${VERL_RECIPE_DIR:?set CG_VERL_RECIPE_DIR (or VERL_RECIPE_DIR) to a compatible checkout}"

MODEL_NAME="${MODEL_NAME:-Qwen3-8B-Base}"
SOURCE_MODEL_PATH="${SOURCE_MODEL_PATH:-$CG_MODEL_ROOT/$MODEL_NAME}"
ARM="${ARM:-full_dapo}"
DATA_DIR="${DATA_DIR:-$CG_DATA_ROOT/stage3_dapo_full_verified}"
PYTHON="${PYTHON:-python}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-32}"
ROLLOUT_N="${ROLLOUT_N:-8}"
if [[ -z "${MAX_RESPONSE_LENGTH:-}" ]]; then
  [[ "$MODEL_NAME" == "Qwen3-8B-Base" ]] && MAX_RESPONSE_LENGTH=3072 || MAX_RESPONSE_LENGTH=2048
fi
CHECKPOINT_BASE="${CHECKPOINT_BASE:-$CG_OUTPUT_ROOT/stage05_legacy_taco_dapo_ablation/$MODEL_NAME/$ARM}"

if [[ ! -f "$DATA_DIR/manifest.json" || ! -f "$DATA_DIR/train.parquet" || ! -f "$DATA_DIR/val.parquet" ]]; then
  echo "[fatal] Stage 05 dataset is incomplete: $DATA_DIR" >&2
  exit 2
fi
command -v "$PYTHON" >/dev/null 2>&1 || { echo "[fatal] python command not found: $PYTHON" >&2; exit 2; }

TRAIN_ROWS=$(
  "$PYTHON" - "$DATA_DIR/train.parquet" <<'PY'
import sys
import pyarrow.parquet as pq
print(pq.ParquetFile(sys.argv[1]).metadata.num_rows)
PY
)
TOTAL_STEPS="${TOTAL_STEPS:-$(( (TRAIN_ROWS + TRAIN_BATCH_SIZE - 1) / TRAIN_BATCH_SIZE ))}"

echo "formal_train_rows=$TRAIN_ROWS"
echo "formal_total_step_ceiling=$TOTAL_STEPS"

export CG_DATA_ROOT CG_MODEL_ROOT CG_OUTPUT_ROOT CG_WORK_ROOT VERL_RECIPE_DIR
export SOURCE_MODEL_PATH MODEL_NAME ARM DATA_DIR MAX_RESPONSE_LENGTH TOTAL_STEPS
export TRAIN_BATCH_SIZE GEN_BATCH_SIZE ROLLOUT_N
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.48}"
export FILTER_GROUPS_ENABLE=True
export FILTER_GROUPS_METRIC=seq_reward
export FILTER_GROUPS_MAX_GEN_BATCHES=0
export OVERLONG_REWARD_ENABLE=False
export OVERLONG_REWARD_BUFFER=0
export OVERLONG_REWARD_PENALTY=0.0
export LR="${LR:-1e-6}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
export LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-10}"
export LOSS_AGG_MODE=token-mean
export CLIP_RATIO_C=10.0
export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
export ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-1.0}"
export DATA_SHUFFLE=True
export DATA_SEED="${DATA_SEED:-20260609}"
export VAL_BEFORE_TRAIN=True
export TEST_FREQ="${TEST_FREQ:-10}"
export SAVE_FREQ="${SAVE_FREQ:-50}"
export MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-2}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-$CHECKPOINT_BASE/resume}"
export ADAPTER_DIR="${ADAPTER_DIR:-$CHECKPOINT_BASE}"
export RESUME_MODE=auto
export VAL_DO_SAMPLE=False
export VAL_ROLLOUT_N=1
export REWARD_EXEC_TIMEOUT="${REWARD_EXEC_TIMEOUT:-180.0}"
export REWARD_PER_TEST_TIMEOUT="${REWARD_PER_TEST_TIMEOUT:-1.0}"
export REWARD_MEMORY_LIMIT_GIB="${REWARD_MEMORY_LIMIT_GIB:-2.0}"
export REWARD_OUTPUT_LIMIT_MIB="${REWARD_OUTPUT_LIMIT_MIB:-16.0}"
export REWARD_WORKERS="${REWARD_WORKERS:-64}"
export RAY_NUM_CPUS="${RAY_NUM_CPUS:-80}"
export RAY_OBJECT_STORE_MEMORY="${RAY_OBJECT_STORE_MEMORY:-68719476736}"

bash "$SCRIPT_DIR/run_job.sh"
