#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
: "${CG_DATA_ROOT:?set CG_DATA_ROOT to the external dataset root}"
: "${CG_MODEL_ROOT:?set CG_MODEL_ROOT to the external model root}"
: "${CG_OUTPUT_ROOT:?set CG_OUTPUT_ROOT to the external output root}"
: "${CG_WORK_ROOT:?set CG_WORK_ROOT to the external scratch/work root}"
: "${CG_DATA_TEST_ROOT:?set CG_DATA_TEST_ROOT to the external data_test root}"
WORK="${WORK:-$CG_OUTPUT_ROOT/stage02_kodcode_sft_diagnosis}"
SCRATCH="${SCRATCH:-$CG_WORK_ROOT/stage02_kodcode_sft_diagnosis}"
PYTHON="${PYTHON:-python}"
SEED="${SEED:-20260724}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
FORMAL_MAX_NEW_TOKENS="${FORMAL_MAX_NEW_TOKENS:-2048}"
FORMAL_CHOICE_BATCH_SIZE="${FORMAL_CHOICE_BATCH_SIZE:-16}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
SCRIPTS="$SCRIPT_DIR"
SOURCE_JSONL="${SOURCE_JSONL:-$CG_DATA_ROOT/kodcode4o_r1_clean38k/train.jsonl}"
SOURCE_PARQUET="${SOURCE_PARQUET:-$CG_DATA_ROOT/kodcode4o_r1_clean38k/train.parquet}"
DATA="${DATA:-$WORK/data}"
CHECKPOINTS="$WORK/checkpoints"
RESULTS="$WORK/results"
LOGS="$WORK/logs"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$SCRATCH/tmp"
export TMPDIR="${TMPDIR:-$SCRATCH/tmp}"

declare -A MODELS=(
  [qwen25]="$CG_MODEL_ROOT/Qwen2.5-7B-Instruct"
  [qwen3]="$CG_MODEL_ROOT/Qwen3-8B-Base"
  [llama31]="$CG_MODEL_ROOT/Llama-3.1-8B-Instruct"
  [gemma2]="$CG_MODEL_ROOT/Gemma-2-9B-Instruct"
)
MODES=(plain native)
FORMAL_TASKS="arc_challenge,scienceqa,legalbench,gsm8k,math500_medium,math500_high_level,finqa,medcalc,humaneval,mbpp_plus,mbpp_simple"
TRAIN_FILE="$DATA/train_full_seed${SEED}.jsonl"

mkdir -p "$DATA" "$CHECKPOINTS" "$RESULTS" "$LOGS"
exec > >(tee -a "$LOGS/run_all.log") 2>&1
echo "[start] $(date --iso-8601=seconds) host=$(hostname)"
nvidia-smi
gpu_count="$(nvidia-smi -L | wc -l | tr -d ' ')"
if [[ "$gpu_count" -ne 2 && "$gpu_count" -ne 4 ]]; then
  echo "[fatal] expected 2 or 4 visible GPUs, got $gpu_count" >&2
  exit 2
fi

GPU_MONITOR_PID=""
cleanup() {
  if [[ -n "$GPU_MONITOR_PID" ]]; then
    kill "$GPU_MONITOR_PID" 2>/dev/null || true
    wait "$GPU_MONITOR_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM
nvidia-smi \
  --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,power.draw \
  --format=csv -l 2 > "$LOGS/gpu_monitor.csv" 2>&1 &
GPU_MONITOR_PID=$!

for key in qwen25 qwen3 llama31 gemma2; do
  [[ -d "${MODELS[$key]}" ]] || {
    echo "[fatal] missing model: ${MODELS[$key]}" >&2
    exit 2
  }
done
if [[ ! -s "$DATA/data_manifest.json" || ! -s "$TRAIN_FILE" ]]; then
  "$PYTHON" "$SCRIPTS/prepare_data.py" \
    --source-jsonl "$SOURCE_JSONL" \
    --source-parquet "$SOURCE_PARQUET" \
    --output-file "$TRAIN_FILE" \
    --manifest "$DATA/data_manifest.json" \
    --seed "$SEED" \
    --overwrite
fi

if [[ ! -s "$DATA/token_length_audit.json" ]]; then
  "$PYTHON" "$SCRIPTS/audit_token_lengths.py" \
    --train-file "$SOURCE_JSONL" \
    --model "qwen25=${MODELS[qwen25]}" \
    --model "qwen3=${MODELS[qwen3]}" \
    --model "llama31=${MODELS[llama31]}" \
    --model "gemma2=${MODELS[gemma2]}" \
    --output "$DATA/token_length_audit.json"
fi
"$PYTHON" - "$DATA/token_length_audit.json" "$MAX_SEQ_LEN" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
limit = int(sys.argv[2])
expected = {"qwen25", "qwen3", "llama31", "gemma2"}
if set(report.get("models", {})) != expected:
    raise SystemExit(f"incomplete token audit: {set(report.get('models', {}))}")
for model, modes in report["models"].items():
    if set(modes) != {"plain", "native"}:
        raise SystemExit(f"incomplete prompt-mode audit for {model}: {set(modes)}")
    for mode, values in modes.items():
        maximum = int(values["total"]["max"])
        print(f"[length] {model}/{mode} max={maximum} limit={limit}")
        if maximum > limit:
            raise SystemExit(
                f"{model}/{mode} has {maximum} tokens > {limit}; refusing silent truncation"
            )
PY

run_train() {
  local model_key="$1" mode="$2" output_dir="$3" port="$4"
  [[ -s "$output_dir/training_summary.json" ]] && return 0
  mkdir -p "$output_dir"
  "$PYTHON" -m torch.distributed.run --nproc_per_node=2 \
    --master_addr=127.0.0.1 --master_port="$port" \
    "$SCRIPTS/train_sft_lora.py" \
    --model-key "$model_key" \
    --model-path "${MODELS[$model_key]}" \
    --train-file "$TRAIN_FILE" \
    --output-dir "$output_dir" \
    --prompt-mode "$mode" \
    --target-profile attention \
    --learning-rate "$LEARNING_RATE" \
    --seed "$SEED" \
    --max-seq-len "$MAX_SEQ_LEN" \
    --per-device-batch-size 1 \
    --gradient-accumulation-steps 8 \
    --epochs 1.0 \
    --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 \
    --warmup-ratio 0.03 --weight-decay 0.01 \
    --gradient-checkpointing \
    --save-midpoint \
    --resume
}

run_pair() {
  local model_key="$1" mode="$2" port="$3"
  local pair="$model_key/$mode"
  local train_dir="$CHECKPOINTS/full/$pair"
  echo "[pair] START $pair $(date --iso-8601=seconds)"

  run_train "$model_key" "$mode" "$train_dir" "$port"

  local midpoint
  midpoint="$(
    find "$train_dir/trainer_checkpoints" -maxdepth 1 -type d -name 'checkpoint-*' \
      -print 2>/dev/null | sort -V | head -1
  )"
  [[ -n "$midpoint" && -s "$midpoint/adapter_model.safetensors" ]] || {
    echo "[fatal] missing midpoint adapter for $pair" >&2
    return 2
  }
  local final_adapter="$train_dir/final_adapter"
  [[ -s "$final_adapter/adapter_model.safetensors" ]] || {
    echo "[fatal] missing final adapter for $pair" >&2
    return 2
  }
  echo "[pair] DONE $pair $(date --iso-8601=seconds)"
}

run_lane() {
  local visible="$1" port="$2"
  shift 2
  export CUDA_VISIBLE_DEVICES="$visible"
  for model_key in "$@"; do
    for mode in "${MODES[@]}"; do
      run_pair "$model_key" "$mode" "$port"
    done
  done
}

if [[ "$gpu_count" -eq 4 ]]; then
  (run_lane "0,1" 29701 qwen25 qwen3) & lane_a=$!
  (run_lane "2,3" 29711 llama31 gemma2) & lane_b=$!
  wait "$lane_a"
  wait "$lane_b"
else
  run_lane "0,1" 29701 qwen25 qwen3 llama31 gemma2
fi
echo "[phase] all training completed"

run_formal_model() {
  local model_key="$1" gpu="$2"
  export CUDA_VISIBLE_DEVICES="$gpu"
  for mode in "${MODES[@]}"; do
    local pair="$model_key/$mode"
    local eval_prompt_mode="$mode"
    [[ "$mode" == native ]] && eval_prompt_mode="chat"
    local train_dir="$CHECKPOINTS/full/$pair"
    local midpoint
    midpoint="$(
      find "$train_dir/trainer_checkpoints" -maxdepth 1 -type d -name 'checkpoint-*' \
        -print 2>/dev/null | sort -V | head -1
    )"
    local final_adapter="$train_dir/final_adapter"
    for variant in base half_epoch final; do
      local output="$RESULTS/formal/$pair/$variant"
      mkdir -p "$output"
      local adapter_args=()
      [[ "$variant" == half_epoch ]] && adapter_args=(--adapter-path "$midpoint")
      [[ "$variant" == final ]] && adapter_args=(--adapter-path "$final_adapter")
      "$PYTHON" "$REPO_ROOT/evaluation/paper_suite.py" \
        --data-root "$CG_DATA_TEST_ROOT" \
        --model-name "diagnise2_${model_key}_${mode}_${variant}" \
        --base-model-path "${MODELS[$model_key]}" \
        --output-dir "$output" \
        --seed "$SEED" \
        --prompt-mode "$eval_prompt_mode" \
        --temperature 0.2 --top-p 0.95 \
        --max-new-tokens-code "$FORMAL_MAX_NEW_TOKENS" \
        --choice-batch-size "$FORMAL_CHOICE_BATCH_SIZE" \
        --generation-batch-size 8 \
        --tasks "$FORMAL_TASKS" \
        --limit 0 \
        --skip-healthbench-generation \
        --resume \
        "${adapter_args[@]}"
    done
  done
}

if [[ "$gpu_count" -eq 4 ]]; then
  (run_formal_model qwen25 0) & eval_a=$!
  (run_formal_model qwen3 1) & eval_b=$!
  (run_formal_model llama31 2) & eval_c=$!
  (run_formal_model gemma2 3) & eval_d=$!
  wait "$eval_a" "$eval_b" "$eval_c" "$eval_d"
else
  (run_formal_model qwen25 0) & eval_a=$!
  (run_formal_model qwen3 1) & eval_b=$!
  wait "$eval_a" "$eval_b"
  (run_formal_model llama31 0) & eval_c=$!
  (run_formal_model gemma2 1) & eval_d=$!
  wait "$eval_c" "$eval_d"
fi

"$PYTHON" "$SCRIPTS/summarize_results.py" --work-dir "$WORK"
"$PYTHON" "$SCRIPTS/select_paper_rows.py" \
  --results-csv "$RESULTS/results.csv" \
  --output "$RESULTS/paper_rows.csv" \
  --manifest "$RESULTS/paper_rows_manifest.json"
echo "[done] $(date --iso-8601=seconds)"
