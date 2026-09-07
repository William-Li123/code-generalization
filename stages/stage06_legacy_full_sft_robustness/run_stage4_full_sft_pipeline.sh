#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
: "${CG_DATA_ROOT:?set CG_DATA_ROOT to the external dataset root}"
: "${CG_MODEL_ROOT:?set CG_MODEL_ROOT to the external model root}"
: "${CG_OUTPUT_ROOT:?set CG_OUTPUT_ROOT to the external output root}"
CG_WORK_ROOT="${CG_WORK_ROOT:-$CG_OUTPUT_ROOT/.work}"
PYTHON="${PYTHON:-python}"
EXP_NAME="${EXP_NAME:-stage4_taco_full_sft}"
MODEL_NAME="${MODEL_NAME:-Qwen2.5-7B-Instruct}"
MODEL_PATH="${MODEL_PATH:-$CG_MODEL_ROOT/$MODEL_NAME}"
SEED="${SEED:-20260603}"
SOURCE_DATA_DIR="${SOURCE_DATA_DIR:-$CG_DATA_ROOT/stage2_taco_sft_clean_prompt}"
DATA_DIR="${DATA_DIR:-$SOURCE_DATA_DIR}"
MODEL_CKPT_ROOT="${MODEL_CKPT_ROOT:-$CG_OUTPUT_ROOT/checkpoints/sft_full/$EXP_NAME/$MODEL_NAME}"
MODEL_LOG_ROOT="${MODEL_LOG_ROOT:-$CG_WORK_ROOT/logs/$EXP_NAME/$MODEL_NAME}"

MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
EPOCHS="${EPOCHS:-1.0}"
MAX_STEPS="${MAX_STEPS:-0}"
LR="${LR:-2e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
SAVE_DTYPE="${SAVE_DTYPE:-bf16}"
OVERWRITE="${OVERWRITE:-0}"
SMOKE="${SMOKE:-0}"
SMOKE_ROWS="${SMOKE_ROWS:-64}"
SMOKE_STEPS="${SMOKE_STEPS:-8}"
DEMO="${DEMO:-0}"
DEMO_ROWS="${DEMO_ROWS:-512}"
DEMO_STEPS="${DEMO_STEPS:-40}"
ONLY_ARMS="${ONLY_ARMS:-full_sft}"
FSDP="${FSDP:-full_shard auto_wrap}"
FSDP_LAYER_CLS="${FSDP_LAYER_CLS:-}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

cd "$REPO_ROOT"

echo "[stage4-full-sft] repo_root=$REPO_ROOT"
echo "[stage4-full-sft] python=$PYTHON"
echo "[stage4-full-sft] model_name=$MODEL_NAME"
echo "[stage4-full-sft] model_path=$MODEL_PATH"
echo "[stage4-full-sft] seed=$SEED"
echo "[stage4-full-sft] source_data_dir=$SOURCE_DATA_DIR"
echo "[stage4-full-sft] data_dir=$DATA_DIR"
echo "[stage4-full-sft] ckpt_root=$MODEL_CKPT_ROOT"
echo "[stage4-full-sft] log_root=$MODEL_LOG_ROOT"
echo "[stage4-full-sft] smoke=$SMOKE demo=$DEMO only_arms=${ONLY_ARMS:-<all>}"
echo "[stage4-full-sft] save_dtype=$SAVE_DTYPE"
echo "[stage4-full-sft] hostname=$(hostname)"
nvidia-smi || true

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

IFS=',' read -r -a ARMS <<< "$ONLY_ARMS"

if [[ "$SMOKE" == "1" || "$SMOKE" == "true" ]]; then
  MAX_SEQ_LEN="${SMOKE_MAX_SEQ_LEN:-1024}"
  MAX_STEPS="$SMOKE_STEPS"
  LOGGING_STEPS=1
  DATA_DIR="$CG_WORK_ROOT/data/${EXP_NAME}_smoke_${MODEL_NAME}"
  MODEL_CKPT_ROOT="$CG_OUTPUT_ROOT/checkpoints/sft_full/${EXP_NAME}_smoke/$MODEL_NAME"
  MODEL_LOG_ROOT="$CG_WORK_ROOT/logs/${EXP_NAME}_smoke/$MODEL_NAME"
  echo "[stage4-full-sft] preparing smoke data rows=$SMOKE_ROWS"
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

if [[ "$DEMO" == "1" || "$DEMO" == "true" ]]; then
  MAX_SEQ_LEN="${DEMO_MAX_SEQ_LEN:-4096}"
  MAX_STEPS="$DEMO_STEPS"
  DATA_DIR="$CG_WORK_ROOT/data/${EXP_NAME}_demo_${MODEL_NAME}"
  MODEL_CKPT_ROOT="$CG_OUTPUT_ROOT/checkpoints/sft_full/${EXP_NAME}_demo/$MODEL_NAME"
  MODEL_LOG_ROOT="$CG_WORK_ROOT/logs/${EXP_NAME}_demo/$MODEL_NAME"
  echo "[stage4-full-sft] preparing demo data rows=$DEMO_ROWS"
  rm -rf "$DATA_DIR"
  mkdir -p "$DATA_DIR"
  "$PYTHON" - "$SOURCE_DATA_DIR" "$DATA_DIR" "$DEMO_ROWS" "${ARMS[@]}" <<'PY'
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
  echo "[stage4-full-sft] overwriting target checkpoint/log roots"
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

for arm in "${ARMS[@]}"; do
  train_file="$DATA_DIR/${arm}.jsonl"
  ckpt_dir="$MODEL_CKPT_ROOT/${arm}"
  log_dir="$MODEL_LOG_ROOT/${arm}"
  mkdir -p "$ckpt_dir" "$log_dir"

  if [[ "$OVERWRITE" != "1" && "$OVERWRITE" != "true" && -s "$ckpt_dir/run_manifest.json" ]]; then
    echo "[stage4-full-sft] SKIP existing arm=$arm"
    continue
  fi

  echo "[stage4-full-sft] START arm=$arm"
  date '+%F %T'
  "$PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" \
    "$SCRIPT_DIR/train_stage4_taco_full_sft.py" \
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
      --max-steps "$MAX_STEPS" \
      --learning-rate "$LR" \
      --weight-decay "$WEIGHT_DECAY" \
      --logging-steps "$LOGGING_STEPS" \
      --fsdp "$FSDP" \
      ${FSDP_LAYER_CLS:+--fsdp-layer-cls "$FSDP_LAYER_CLS"} \
      --save-dtype "$SAVE_DTYPE" \
      --gradient-checkpointing \
      > "$log_dir/train_stdout.log" 2> "$log_dir/train_stderr.log"
  echo "[stage4-full-sft] DONE arm=$arm"
  date '+%F %T'
done

"$PYTHON" - "$MODEL_CKPT_ROOT" "$MODEL_LOG_ROOT" "$SEED" "${ARMS[@]}" <<'PY'
import json
import sys
from pathlib import Path

ckpt_root = Path(sys.argv[1])
log_root = Path(sys.argv[2])
seed = int(sys.argv[3])
arms = sys.argv[4:]
rows = []
for arm in arms:
    result_path = log_root / arm / "training_result.json"
    row = {
        "arm": arm,
        "seed": seed,
        "manifest_exists": (ckpt_root / arm / "run_manifest.json").exists(),
        "safetensor_count": len(list((ckpt_root / arm).glob("*.safetensors"))),
    }
    if result_path.exists():
        try:
            row.update(json.loads(result_path.read_text(encoding="utf-8")))
        except Exception as exc:
            row["result_read_error"] = repr(exc)
    rows.append(row)
summary = {"seed": seed, "checkpoint_root": str(ckpt_root), "log_root": str(log_root), "runs": rows}
(log_root / "stage4_full_sft_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

echo "[stage4-full-sft] complete"
