#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STAGE_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd -- "$STAGE_ROOT/../.." && pwd)"
: "${CG_MODEL_ROOT:?set CG_MODEL_ROOT to the external model root}"
: "${CG_OUTPUT_ROOT:?set CG_OUTPUT_ROOT to the external output root}"
: "${CG_WORK_ROOT:?set CG_WORK_ROOT to the external scratch/work root}"
WORK="${WORK:-$CG_OUTPUT_ROOT/stage03_kodcode_dapo_diagnosis/rl10k}"
PYTHON="${PYTHON:-python}"
MODEL_KEY="${MODEL_KEY:?MODEL_KEY is required}"
SEED="${SEED:-20260725}"
DATA_DIR="${DATA_DIR:-$WORK/data}"

case "$MODEL_KEY" in
  qwen25)
    MODEL_NAME="Qwen2.5-7B-Instruct"
    TRAIN_PROMPT_MODE="native"
    MAX_RESPONSE_LENGTH=2048
    GPU_MEMORY_UTILIZATION=0.38
    MICRO_BATCH=4
    TOKENIZER_CHAT_TEMPLATE_FILE=""
    EXTRA_ENV=()
    ;;
  qwen3)
    MODEL_NAME="Qwen3-8B-Base"
    TRAIN_PROMPT_MODE="plain"
    MAX_RESPONSE_LENGTH=3072
    GPU_MEMORY_UTILIZATION=0.34
    MICRO_BATCH=4
    TOKENIZER_CHAT_TEMPLATE_FILE="$STAGE_ROOT/plain_chat_template.jinja"
    EXTRA_ENV=()
    ;;
  llama31)
    MODEL_NAME="Llama-3.1-8B-Instruct"
    TRAIN_PROMPT_MODE="native"
    MAX_RESPONSE_LENGTH=3072
    GPU_MEMORY_UTILIZATION=0.32
    MICRO_BATCH=4
    TOKENIZER_CHAT_TEMPLATE_FILE=""
    EXTRA_ENV=()
    ;;
  gemma2)
    MODEL_NAME="Gemma-2-9B-Instruct"
    TRAIN_PROMPT_MODE="native"
    MAX_RESPONSE_LENGTH=2048
    GPU_MEMORY_UTILIZATION=0.30
    MICRO_BATCH=4
    TOKENIZER_CHAT_TEMPLATE_FILE=""
    EXTRA_ENV=(
      VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1
      ROLLOUT_HF_OVERRIDES_ROPE_THETA=10000.0
    )
    ;;
  *)
    echo "unknown MODEL_KEY=$MODEL_KEY" >&2
    exit 2
    ;;
esac

if [[ "$TRAIN_PROMPT_MODE" == "plain" && -z "$TOKENIZER_CHAT_TEMPLATE_FILE" ]]; then
  echo "plain training requires an explicit compatibility template" >&2
  exit 2
fi
if [[ "$TRAIN_PROMPT_MODE" == "native" && -n "$TOKENIZER_CHAT_TEMPLATE_FILE" ]]; then
  echo "native training must use the model tokenizer template unchanged" >&2
  exit 2
fi

SOURCE_MODEL_PATH="$CG_MODEL_ROOT/$MODEL_NAME"
CHECKPOINT_BASE="$WORK/checkpoints/$MODEL_KEY/full_dapo"
DONE_MARKER="$CHECKPOINT_BASE/.training_complete"
if [[ -f "$DONE_MARKER" ]]; then
  echo "[skip] $MODEL_KEY already complete: $DONE_MARKER"
  exit 0
fi
for required in \
  "$SOURCE_MODEL_PATH/config.json" \
  "$DATA_DIR/train.parquet" \
  "$DATA_DIR/val.parquet" \
  "$DATA_DIR/manifest.json"; do
  [[ -f "$required" ]] || { echo "missing required file: $required" >&2; exit 3; }
done

TRAIN_ROWS="$("$PYTHON" - "$DATA_DIR/train.parquet" <<'PY'
import pyarrow.parquet as pq
import sys
print(pq.ParquetFile(sys.argv[1]).metadata.num_rows)
PY
)"
TRAIN_BATCH_SIZE=32
TOTAL_STEPS=$(( (TRAIN_ROWS + TRAIN_BATCH_SIZE - 1) / TRAIN_BATCH_SIZE ))
RUN_NAME="${RUN_NAME:-${MODEL_KEY}-full-dapo-2gpu-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="$WORK/output/$RUN_NAME"
SYSTEM_LOG="$WORK/logs/$RUN_NAME.system.log"
mkdir -p "$CHECKPOINT_BASE" "$RUN_DIR" "$WORK/logs"

echo "[train] model=$MODEL_KEY prompt_mode=$TRAIN_PROMPT_MODE rows=$TRAIN_ROWS ceiling=$TOTAL_STEPS response=$MAX_RESPONSE_LENGTH"
if [[ "${DRY_RUN:-0}" == "1" || "${DRY_RUN:-false}" == "true" ]]; then
  printf '[dry-run] model=%s source=%s data=%s rows=%s step_ceiling=%s response=%s util=%s micro_batch=%s prompt_mode=%s template=%s\n' \
    "$MODEL_KEY" "$SOURCE_MODEL_PATH" "$DATA_DIR" "$TRAIN_ROWS" "$TOTAL_STEPS" \
    "$MAX_RESPONSE_LENGTH" "$GPU_MEMORY_UTILIZATION" "$MICRO_BATCH" \
    "$TRAIN_PROMPT_MODE" \
    "${TOKENIZER_CHAT_TEMPLATE_FILE:-native}"
  exit 0
fi
printf 'model_key=%s\nmodel_name=%s\ntraining_prompt_mode=%s\ntokenizer_template=%s\n' \
  "$MODEL_KEY" "$MODEL_NAME" "$TRAIN_PROMPT_MODE" \
  "${TOKENIZER_CHAT_TEMPLATE_FILE:-native tokenizer template}" \
  > "$RUN_DIR/prompt_contract.txt"
env \
  REPO_ROOT="$REPO_ROOT" \
  SOURCE_MODEL_PATH="$SOURCE_MODEL_PATH" \
  MODEL_NAME="$MODEL_NAME" \
  DATA_DIR="$DATA_DIR" \
  CHECKPOINT_DIR="$CHECKPOINT_BASE/resume" \
  ADAPTER_DIR="$CHECKPOINT_BASE" \
  RUN_NAME="$RUN_NAME" \
  RUN_DIR="$RUN_DIR" \
  SYSTEM_LOG="$SYSTEM_LOG" \
  EXPECTED_GPUS=2 \
  N_GPUS=2 \
  CUDA_VISIBLE_DEVICES=0,1 \
  LOCAL_MODEL_ROOT="$CG_WORK_ROOT/stage03_kodcode_dapo_diagnosis/model_cache/rl10k_${MODEL_KEY}" \
  TOKENIZER_CHAT_TEMPLATE_FILE="$TOKENIZER_CHAT_TEMPLATE_FILE" \
  TOTAL_STEPS="$TOTAL_STEPS" \
  TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE" \
  GEN_BATCH_SIZE=32 \
  ROLLOUT_N=8 \
  MAX_RESPONSE_LENGTH="$MAX_RESPONSE_LENGTH" \
  MAX_PROMPT_LENGTH=4096 \
  GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" \
  USE_DYNAMIC_BSZ=False \
  PPO_MICRO_BATCH_SIZE_PER_GPU="$MICRO_BATCH" \
  LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="$MICRO_BATCH" \
  PPO_MAX_TOKEN_LEN=4096 \
  LOG_PROB_MAX_TOKEN_LEN=8192 \
  MAX_NUM_BATCHED_TOKENS=24576 \
  MAX_NUM_SEQS=32 \
  FILTER_GROUPS_ENABLE=True \
  FILTER_GROUPS_METRIC=seq_reward \
  FILTER_GROUPS_MAX_GEN_BATCHES=6 \
  PARAM_OFFLOAD=False \
  OPTIMIZER_OFFLOAD=False \
  USE_ORIG_PARAMS=True \
  LAYERED_SUMMON=True \
  MULTI_STAGE_WAKE_UP=True \
  UPDATE_WEIGHTS_BUCKET_MEGABYTES=512 \
  LR=1e-6 \
  WEIGHT_DECAY=0.1 \
  LR_WARMUP_STEPS=10 \
  LOSS_AGG_MODE=token-mean \
  CLIP_RATIO_C=10.0 \
  ROLLOUT_TEMPERATURE=1.0 \
  ROLLOUT_TOP_P=1.0 \
  DATA_SHUFFLE=True \
  DATA_SEED="$SEED" \
  VAL_BEFORE_TRAIN=True \
  TEST_FREQ=50 \
  SAVE_FREQ=50 \
  MAX_ACTOR_CKPT_TO_KEEP=1 \
  RESUME_MODE=auto \
  VAL_DO_SAMPLE=False \
  VAL_ROLLOUT_N=1 \
  REWARD_PATH="$STAGE_ROOT/shared/kodcode_reward.py" \
  REWARD_WORKERS=8 \
  RAY_NUM_CPUS=40 \
  RAY_OBJECT_STORE_MEMORY=34359738368 \
  REWARD_EXEC_TIMEOUT=180.0 \
  REWARD_PER_TEST_TIMEOUT=1.0 \
  REWARD_MEMORY_LIMIT_GIB=2.0 \
  REWARD_OUTPUT_LIMIT_MIB=16.0 \
  VLLM_USE_V1=1 \
  "${EXTRA_ENV[@]}" \
  bash "$STAGE_ROOT/shared/run_job.sh"

LATEST_FILE="$CHECKPOINT_BASE/latest_adapter.txt"
[[ -s "$LATEST_FILE" ]] || { echo "missing latest adapter pointer: $LATEST_FILE" >&2; exit 4; }
LATEST_ADAPTER="$(cat "$LATEST_FILE")"
[[ "$LATEST_ADAPTER" = /* ]] || LATEST_ADAPTER="$CHECKPOINT_BASE/$LATEST_ADAPTER"
for required in adapter_config.json adapter_model.safetensors; do
  [[ -f "$LATEST_ADAPTER/$required" ]] || { echo "invalid final adapter: $LATEST_ADAPTER/$required" >&2; exit 5; }
done
printf 'model=%s\ntrain_rows=%s\ntotal_step_ceiling=%s\nlatest_adapter=%s\ncompleted_at=%s\n' \
  "$MODEL_KEY" "$TRAIN_ROWS" "$TOTAL_STEPS" "$LATEST_ADAPTER" "$(date -Iseconds)" \
  > "$DONE_MARKER.tmp"
mv "$DONE_MARKER.tmp" "$DONE_MARKER"
rm -rf "$CG_WORK_ROOT/stage03_kodcode_dapo_diagnosis/model_cache/rl10k_${MODEL_KEY}"
echo "[complete] $MODEL_KEY adapter=$LATEST_ADAPTER"
