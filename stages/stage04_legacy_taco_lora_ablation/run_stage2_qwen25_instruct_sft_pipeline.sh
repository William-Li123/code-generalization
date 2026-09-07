#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
: "${CG_DATA_ROOT:?set CG_DATA_ROOT to the external dataset root}"
: "${CG_MODEL_ROOT:?set CG_MODEL_ROOT to the external model root}"
: "${CG_OUTPUT_ROOT:?set CG_OUTPUT_ROOT to the external output root}"
CG_WORK_ROOT="${CG_WORK_ROOT:-$CG_OUTPUT_ROOT/.work}"
PYTHON="${PYTHON:-python}"
EXP_NAME="${EXP_NAME:-stage2_taco_sft_clean_prompt}"
MODEL_NAME="${MODEL_NAME:-Qwen2.5-7B-Instruct}"
MODEL_PATH="${MODEL_PATH:-$CG_MODEL_ROOT/$MODEL_NAME}"
SEED="${SEED:-20260603}"
USER_MAX_SEQ_LEN="${MAX_SEQ_LEN:-}"
USER_GRAD_ACCUM="${GRAD_ACCUM:-}"
USER_LOGGING_STEPS="${LOGGING_STEPS:-}"
USER_DATA_DIR="${DATA_DIR:-}"
USER_MODEL_CKPT_ROOT="${MODEL_CKPT_ROOT:-}"
USER_MODEL_LOG_ROOT="${MODEL_LOG_ROOT:-}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
EPOCHS="${EPOCHS:-1.0}"
LR="${LR:-1e-4}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
OVERWRITE="${OVERWRITE:-1}"
SMOKE="${SMOKE:-0}"
SMOKE_ROWS="${SMOKE_ROWS:-8}"
ONLY_ARMS="${ONLY_ARMS:-}"

SOURCE_DATA_DIR="${SOURCE_DATA_DIR:-$CG_DATA_ROOT/$EXP_NAME}"
MODEL_CKPT_ROOT="${MODEL_CKPT_ROOT:-$CG_OUTPUT_ROOT/checkpoints/sft_lora/$EXP_NAME/$MODEL_NAME}"
MODEL_LOG_ROOT="${MODEL_LOG_ROOT:-$CG_WORK_ROOT/logs/$EXP_NAME/$MODEL_NAME}"
DATA_DIR="${DATA_DIR:-$SOURCE_DATA_DIR}"

if [[ "$SMOKE" == "1" || "$SMOKE" == "true" ]]; then
  [[ -z "$USER_MAX_SEQ_LEN" ]] && MAX_SEQ_LEN=1024
  [[ -z "$USER_GRAD_ACCUM" ]] && GRAD_ACCUM=1
  [[ -z "$USER_LOGGING_STEPS" ]] && LOGGING_STEPS=1
  [[ -z "$USER_MODEL_CKPT_ROOT" ]] && MODEL_CKPT_ROOT="$CG_OUTPUT_ROOT/checkpoints/sft_lora/${EXP_NAME}_smoke/$MODEL_NAME"
  [[ -z "$USER_MODEL_LOG_ROOT" ]] && MODEL_LOG_ROOT="$CG_WORK_ROOT/logs/${EXP_NAME}_smoke/$MODEL_NAME"
  [[ -z "$USER_DATA_DIR" ]] && DATA_DIR="$CG_WORK_ROOT/data/${EXP_NAME}_smoke_${MODEL_NAME}"
fi

cd "$REPO_ROOT"

echo "[qwen25-instruct-sft] repo_root=$REPO_ROOT"
echo "[qwen25-instruct-sft] python=$PYTHON"
echo "[qwen25-instruct-sft] model_name=$MODEL_NAME"
echo "[qwen25-instruct-sft] model_path=$MODEL_PATH"
echo "[qwen25-instruct-sft] seed=$SEED"
echo "[qwen25-instruct-sft] source_data_dir=$SOURCE_DATA_DIR"
echo "[qwen25-instruct-sft] data_dir=$DATA_DIR"
echo "[qwen25-instruct-sft] ckpt_root=$MODEL_CKPT_ROOT"
echo "[qwen25-instruct-sft] log_root=$MODEL_LOG_ROOT"
echo "[qwen25-instruct-sft] smoke=$SMOKE only_arms=${ONLY_ARMS:-<all>}"
echo "[qwen25-instruct-sft] hostname=$(hostname)"
nvidia-smi || true

GPU_COUNT="${GPU_COUNT:-$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')}"
if [[ -z "$GPU_COUNT" || "$GPU_COUNT" == "0" ]]; then
  GPU_COUNT=4
fi
echo "[qwen25-instruct-sft] gpu_count=$GPU_COUNT"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "[fatal] python command not found: $PYTHON" >&2
  exit 2
fi
if [[ ! -d "$MODEL_PATH" ]]; then
  echo "[fatal] model path missing: $MODEL_PATH" >&2
  exit 2
fi
if [[ ! -d "$SOURCE_DATA_DIR" ]]; then
  echo "[fatal] source data dir missing: $SOURCE_DATA_DIR" >&2
  exit 2
fi

ARMS=(
  "full_sft"
  "full_sft_14043"
  "no_math_number_theory"
  "no_data_structure"
  "no_dynamic_programming"
  "no_greedy_search"
  "no_implementation_simulation"
  "no_graph"
  "no_other_algorithm"
)

if [[ -n "$ONLY_ARMS" ]]; then
  IFS=',' read -r -a ARMS <<< "$ONLY_ARMS"
fi

if [[ "$SMOKE" == "1" || "$SMOKE" == "true" ]]; then
  echo "[qwen25-instruct-sft] preparing smoke data rows=$SMOKE_ROWS"
  rm -rf "$DATA_DIR"
  mkdir -p "$DATA_DIR"
  "$PYTHON" - "$SOURCE_DATA_DIR" "$DATA_DIR" "$SMOKE_ROWS" "${ARMS[@]}" <<'PY'
import sys
from pathlib import Path

source = Path(sys.argv[1])
target = Path(sys.argv[2])
rows = int(sys.argv[3])
arms = sys.argv[4:]
target.mkdir(parents=True, exist_ok=True)
for arm in arms:
    src = source / f"{arm}.jsonl"
    dst = target / f"{arm}.jsonl"
    if not src.exists():
        raise FileNotFoundError(src)
    with src.open(encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for index, line in enumerate(fin):
            if index >= rows:
                break
            fout.write(line)
PY
fi

if [[ "$OVERWRITE" == "1" || "$OVERWRITE" == "true" ]]; then
  echo "[qwen25-instruct-sft] overwriting target checkpoint/log roots"
  [[ "$MODEL_CKPT_ROOT" == "$CG_OUTPUT_ROOT/"* ]] || { echo "[fatal] unsafe ckpt root: $MODEL_CKPT_ROOT" >&2; exit 2; }
  [[ "$MODEL_LOG_ROOT" == "$CG_WORK_ROOT/"* ]] || { echo "[fatal] unsafe log root: $MODEL_LOG_ROOT" >&2; exit 2; }
  rm -rf "$MODEL_CKPT_ROOT" "$MODEL_LOG_ROOT"
fi
mkdir -p "$MODEL_CKPT_ROOT" "$MODEL_LOG_ROOT"

for arm in "${ARMS[@]}"; do
  if [[ ! -s "$DATA_DIR/${arm}.jsonl" ]]; then
    echo "[fatal] train file missing or empty: $DATA_DIR/${arm}.jsonl" >&2
    exit 2
  fi
done

run_one() {
  local gpu="$1"
  local arm="$2"
  local train_file="$DATA_DIR/${arm}.jsonl"
  local ckpt_dir="$MODEL_CKPT_ROOT/${arm}"
  local log_dir="$MODEL_LOG_ROOT/${arm}"
  mkdir -p "$ckpt_dir" "$log_dir"

  if [[ "$OVERWRITE" != "1" && "$OVERWRITE" != "true" && -s "$ckpt_dir/adapter_model.safetensors" ]]; then
    echo "[qwen25-instruct-sft] SKIP existing gpu=$gpu arm=$arm"
    return 0
  fi

  echo "[qwen25-instruct-sft] START gpu=$gpu arm=$arm"
  date '+%F %T'

  CUDA_VISIBLE_DEVICES="$gpu" TOKENIZERS_PARALLELISM=false \
  "$PYTHON" "$SCRIPT_DIR/train_stage2_taco_lora.py" \
    --model-name "$MODEL_NAME" \
    --model-path "$MODEL_PATH" \
    --arm "$arm" \
    --train-file "$train_file" \
    --checkpoint-dir "$ckpt_dir" \
    --log-dir "$log_dir" \
    --seed "$SEED" \
    --max-seq-len "$MAX_SEQ_LEN" \
    --per-device-train-batch-size "$PER_DEVICE_BATCH" \
    --gradient-accumulation-steps "$GRAD_ACCUM" \
    --num-train-epochs "$EPOCHS" \
    --learning-rate "$LR" \
    --lora-r "$LORA_R" \
    --lora-alpha "$LORA_ALPHA" \
    --lora-dropout "$LORA_DROPOUT" \
    --logging-steps "$LOGGING_STEPS" \
    --gradient-checkpointing \
    > "$log_dir/train_stdout.log" 2> "$log_dir/train_stderr.log"

  echo "[qwen25-instruct-sft] DONE gpu=$gpu arm=$arm"
  date '+%F %T'
}

run_queue_for_gpu() {
  local gpu="$1"
  local index="$gpu"
  while [[ "$index" -lt "${#ARMS[@]}" ]]; do
    run_one "$gpu" "${ARMS[$index]}"
    index=$((index + GPU_COUNT))
  done
}

pids=()
for ((gpu=0; gpu<GPU_COUNT; gpu++)); do
  run_queue_for_gpu "$gpu" > "$MODEL_LOG_ROOT/gpu${gpu}_pipeline.log" 2>&1 &
  pids+=("$!")
done
printf '%s\n' "${pids[@]}" > "$MODEL_LOG_ROOT/pipeline_pids.txt"

wait "${pids[@]}"

"$PYTHON" - "$MODEL_CKPT_ROOT" "$MODEL_LOG_ROOT" "$SEED" "$MODEL_NAME" "${ARMS[@]}" <<'PY'
import json
import sys
from pathlib import Path

ckpt_root = Path(sys.argv[1])
log_root = Path(sys.argv[2])
seed = int(sys.argv[3])
model = sys.argv[4]
arms = sys.argv[5:]
rows = []
for arm in arms:
    result_path = log_root / arm / "training_result.json"
    row = {
        "model": model,
        "arm": arm,
        "seed": seed,
        "adapter_exists": (ckpt_root / arm / "adapter_model.safetensors").exists(),
    }
    if result_path.exists():
        try:
            row.update(json.loads(result_path.read_text(encoding="utf-8")))
        except Exception as exc:
            row["result_read_error"] = repr(exc)
    rows.append(row)
summary = {
    "model": model,
    "seed": seed,
    "arm_count": len(arms),
    "checkpoint_root": str(ckpt_root),
    "log_root": str(log_root),
    "runs": rows,
}
(log_root / "sft_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

echo "[qwen25-instruct-sft] complete"
