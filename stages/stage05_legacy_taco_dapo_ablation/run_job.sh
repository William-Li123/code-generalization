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
ARM="${ARM:-full_dapo_12168}"
SEED="${SEED:-20260610}"
if [[ -z "${MAX_RESPONSE_LENGTH:-}" ]]; then
  [[ "$MODEL_NAME" == "Qwen3-8B-Base" ]] && MAX_RESPONSE_LENGTH=3072 || MAX_RESPONSE_LENGTH=2048
fi
TOTAL_STEPS="${TOTAL_STEPS:-}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
ROLLOUT_N="${ROLLOUT_N:-8}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
RUN_NAME="${RUN_NAME:-${MODEL_NAME}__${ARM}__seed${SEED}__${RUN_ID}}"

if [[ "$ARM" == "full_dapo" ]]; then
  DEFAULT_DATA_DIR="$CG_DATA_ROOT/stage3_dapo_full_verified"
else
  DEFAULT_DATA_DIR="$CG_DATA_ROOT/stage3_dapo_ablation_12168/seed_${SEED}/$ARM"
fi
DATA_DIR="${DATA_DIR:-$DEFAULT_DATA_DIR}"
RUN_DIR="${RUN_DIR:-$CG_OUTPUT_ROOT/stage05_legacy_taco_dapo_ablation/$RUN_NAME}"
SYSTEM_LOG="${SYSTEM_LOG:-$CG_WORK_ROOT/logs/stage05_legacy_taco_dapo_ablation/$RUN_NAME.system.log}"
ENV_DIR="${ENV_DIR:-${CG_VERL_ENV:-$CG_WORK_ROOT/verl_env}}"

[[ -d "$SOURCE_MODEL_PATH" ]] || { echo "[fatal] missing model: $SOURCE_MODEL_PATH" >&2; exit 2; }
[[ -f "$DATA_DIR/train.parquet" ]] || { echo "[fatal] missing train data: $DATA_DIR/train.parquet" >&2; exit 2; }
[[ -f "$DATA_DIR/val.parquet" ]] || { echo "[fatal] missing validation data: $DATA_DIR/val.parquet" >&2; exit 2; }
[[ -d "$VERL_RECIPE_DIR" ]] || { echo "[fatal] missing verl-recipe checkout: $VERL_RECIPE_DIR" >&2; exit 2; }

ROW_COUNT_PYTHON="${PYTHON:-$ENV_DIR/bin/python}"
if [[ ! -x "$ROW_COUNT_PYTHON" ]]; then
  ROW_COUNT_PYTHON="${PYTHON:-python}"
fi
TRAIN_ROWS=$(
  "$ROW_COUNT_PYTHON" - "$DATA_DIR/train.parquet" <<'PY'
import sys
import pyarrow.parquet as pq
print(pq.ParquetFile(sys.argv[1]).metadata.num_rows)
PY
)
if [[ -n "${EXPECTED_TRAIN_ROWS:-}" && "$TRAIN_ROWS" != "$EXPECTED_TRAIN_ROWS" ]]; then
  echo "[fatal] expected $EXPECTED_TRAIN_ROWS training rows, found $TRAIN_ROWS in $DATA_DIR" >&2
  exit 2
fi
if [[ -z "$TOTAL_STEPS" ]]; then
  TOTAL_STEPS=$(( (TRAIN_ROWS + TRAIN_BATCH_SIZE - 1) / TRAIN_BATCH_SIZE ))
fi
if [[ "$TOTAL_STEPS" -le 0 ]]; then
  echo "[fatal] TOTAL_STEPS must be positive, got $TOTAL_STEPS" >&2
  exit 2
fi

N_GPUS="${N_GPUS:-4}"
EXPECTED_GPUS="${EXPECTED_GPUS:-$N_GPUS}"
if [[ -n "$EXPECTED_GPUS" ]]; then
  gpu_count="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
  [[ "$gpu_count" == "$EXPECTED_GPUS" ]] || {
    echo "[fatal] expected $EXPECTED_GPUS visible GPUs, found $gpu_count" >&2
    exit 2
  }
fi

mkdir -p "$RUN_DIR" "$(dirname "$SYSTEM_LOG")"
export CG_DATA_ROOT CG_MODEL_ROOT CG_OUTPUT_ROOT CG_WORK_ROOT VERL_RECIPE_DIR
export ENV_DIR MODEL_PATH="$SOURCE_MODEL_PATH" DATA_DIR RUN_DIR
export PROJECT_NAME="${PROJECT_NAME:-stage05_legacy_taco_dapo_ablation}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-$RUN_NAME}"
export DATA_SEED="${DATA_SEED:-$SEED}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-$RUN_DIR/checkpoints}"
export ADAPTER_DIR="${ADAPTER_DIR:-$RUN_DIR/adapters}"
export N_GPUS
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.38}"
export PARAM_OFFLOAD="${PARAM_OFFLOAD:-False}"
export OPTIMIZER_OFFLOAD="${OPTIMIZER_OFFLOAD:-False}"
export USE_ORIG_PARAMS="${USE_ORIG_PARAMS:-True}"
export USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-False}"
if [[ -z "${PPO_MICRO_BATCH_SIZE_PER_GPU:-}" ]]; then
  [[ "$MODEL_NAME" == "Qwen3-8B-Base" ]] && PPO_MICRO_BATCH_SIZE_PER_GPU=4 || PPO_MICRO_BATCH_SIZE_PER_GPU=8
fi
export PPO_MICRO_BATCH_SIZE_PER_GPU
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-$PPO_MICRO_BATCH_SIZE_PER_GPU}"
export PPO_MAX_TOKEN_LEN="${PPO_MAX_TOKEN_LEN:-4096}"
export LOG_PROB_MAX_TOKEN_LEN="${LOG_PROB_MAX_TOKEN_LEN:-8192}"
export LAYERED_SUMMON="${LAYERED_SUMMON:-True}"
export MULTI_STAGE_WAKE_UP="${MULTI_STAGE_WAKE_UP:-True}"
export UPDATE_WEIGHTS_BUCKET_MEGABYTES="${UPDATE_WEIGHTS_BUCKET_MEGABYTES:-512}"
export MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-1}"
export TEST_FREQ="${TEST_FREQ:-50}"
export SAVE_FREQ="${SAVE_FREQ:-50}"
export GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-32}"
export LR="${LR:-1e-6}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
export LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-10}"
export REWARD_WORKERS="${REWARD_WORKERS:-64}"

echo "[stage05] model=$MODEL_NAME arm=$ARM seed=$SEED rows=$TRAIN_ROWS steps=$TOTAL_STEPS data=$DATA_DIR output=$RUN_DIR" | tee -a "$SYSTEM_LOG"
set +e
bash "$REPO_ROOT/dapo/run_training.sh" \
  "$MAX_RESPONSE_LENGTH" \
  "$TOTAL_STEPS" \
  "$TRAIN_BATCH_SIZE" \
  "$ROLLOUT_N" 2>&1 | tee -a "$SYSTEM_LOG"
status=${PIPESTATUS[0]}
set -e
exit "$status"
