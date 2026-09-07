#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
: "${CG_DATA_ROOT:?set CG_DATA_ROOT to the external dataset root}"
: "${CG_MODEL_ROOT:?set CG_MODEL_ROOT to the external model root}"
: "${CG_OUTPUT_ROOT:?set CG_OUTPUT_ROOT to the external output root}"
CG_WORK_ROOT="${CG_WORK_ROOT:-$CG_OUTPUT_ROOT/.work}"
PYTHON="${PYTHON:-python}"
SEED="${SEED:-20260603}"
EXP_NAME="${EXP_NAME:-stage4_taco_full_sft_seed_${SEED}}"
SOURCE_DATA_DIR="${SOURCE_DATA_DIR:-$CG_DATA_ROOT/stage2_taco_sft_clean_prompt}"
MODELS_CSV="${MODELS_CSV:-Qwen2.5-7B-Instruct,Qwen3-8B-Base}"
ARMS_CSV="${ARMS_CSV:-full_sft,full_sft_14043,no_math_number_theory,no_data_structure,no_dynamic_programming,no_greedy_search,no_implementation_simulation,no_graph,no_other_algorithm}"

MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
EPOCHS="${EPOCHS:-1.0}"
MAX_STEPS="${MAX_STEPS:-0}"
LR="${LR:-2e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
LOGGING_STEPS="${LOGGING_STEPS:-25}"
SAVE_DTYPE="${SAVE_DTYPE:-bf16}"
OVERWRITE="${OVERWRITE:-0}"

cd "$REPO_ROOT"

IFS=',' read -r -a MODELS <<< "$MODELS_CSV"
IFS=',' read -r -a ARMS <<< "$ARMS_CSV"

RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
DRIVER_LOG_DIR="$CG_WORK_ROOT/logs/$EXP_NAME/_formal_18arms"
mkdir -p "$DRIVER_LOG_DIR"
DRIVER_MANIFEST="$DRIVER_LOG_DIR/run_${RUN_ID}.json"

echo "[stage4-formal-18] repo_root=$REPO_ROOT"
echo "[stage4-formal-18] exp_name=$EXP_NAME"
echo "[stage4-formal-18] seed=$SEED"
echo "[stage4-formal-18] source_data_dir=$SOURCE_DATA_DIR"
echo "[stage4-formal-18] models=$MODELS_CSV"
echo "[stage4-formal-18] arms=$ARMS_CSV"
echo "[stage4-formal-18] save_dtype=$SAVE_DTYPE overwrite=$OVERWRITE"
echo "[stage4-formal-18] max_seq_len=$MAX_SEQ_LEN per_device_batch=$PER_DEVICE_BATCH grad_accum=$GRAD_ACCUM epochs=$EPOCHS"
echo "[stage4-formal-18] lr=$LR weight_decay=$WEIGHT_DECAY logging_steps=$LOGGING_STEPS"
echo "[stage4-formal-18] hostname=$(hostname)"
nvidia-smi || true

if [[ ! -d "$SOURCE_DATA_DIR" ]]; then
  echo "[fatal] missing source data dir: $SOURCE_DATA_DIR" >&2
  exit 2
fi
for arm in "${ARMS[@]}"; do
  if [[ ! -s "$SOURCE_DATA_DIR/${arm}.jsonl" ]]; then
    echo "[fatal] missing train file: $SOURCE_DATA_DIR/${arm}.jsonl" >&2
    exit 2
  fi
done
for model in "${MODELS[@]}"; do
  if [[ ! -d "$CG_MODEL_ROOT/$model" ]]; then
    echo "[fatal] missing model dir: $CG_MODEL_ROOT/$model" >&2
    exit 2
  fi
done

command -v "$PYTHON" >/dev/null 2>&1 || { echo "[fatal] python command not found: $PYTHON" >&2; exit 2; }
"$PYTHON" - "$DRIVER_MANIFEST" "$CG_OUTPUT_ROOT" "$CG_WORK_ROOT" "$EXP_NAME" "$SEED" "$MODELS_CSV" "$ARMS_CSV" <<'PY'
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
output_root = Path(sys.argv[2])
work_root = Path(sys.argv[3])
exp_name = sys.argv[4]
seed = int(sys.argv[5])
models = sys.argv[6].split(",")
arms = sys.argv[7].split(",")
rows = []
for model in models:
    for arm in arms:
        rows.append(
            {
                "model": model,
                "arm": arm,
                "checkpoint_dir": str(output_root / "checkpoints" / "sft_full" / exp_name / model / arm),
                "log_dir": str(work_root / "logs" / exp_name / model / arm),
                "status": "pending",
            }
        )
manifest.write_text(
    json.dumps(
        {
            "seed": seed,
            "exp_name": exp_name,
            "models": models,
            "arms": arms,
            "runs": rows,
        },
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)
PY

for model in "${MODELS[@]}"; do
  echo "[stage4-formal-18] START model=$model"
  date '+%F %T'
  MODEL_NAME="$model" \
  MODEL_PATH="$CG_MODEL_ROOT/$model" \
  EXP_NAME="$EXP_NAME" \
  SOURCE_DATA_DIR="$SOURCE_DATA_DIR" \
  DATA_DIR="$SOURCE_DATA_DIR" \
  MODEL_CKPT_ROOT="$CG_OUTPUT_ROOT/checkpoints/sft_full/$EXP_NAME/$model" \
  MODEL_LOG_ROOT="$CG_WORK_ROOT/logs/$EXP_NAME/$model" \
  ONLY_ARMS="$ARMS_CSV" \
  SEED="$SEED" \
  MAX_SEQ_LEN="$MAX_SEQ_LEN" \
  PER_DEVICE_BATCH="$PER_DEVICE_BATCH" \
  GRAD_ACCUM="$GRAD_ACCUM" \
  EPOCHS="$EPOCHS" \
  MAX_STEPS="$MAX_STEPS" \
  LR="$LR" \
  WEIGHT_DECAY="$WEIGHT_DECAY" \
  LOGGING_STEPS="$LOGGING_STEPS" \
  SAVE_DTYPE="$SAVE_DTYPE" \
  OVERWRITE="$OVERWRITE" \
  SMOKE=0 \
  DEMO=0 \
  CG_DATA_ROOT="$CG_DATA_ROOT" \
  CG_MODEL_ROOT="$CG_MODEL_ROOT" \
  CG_OUTPUT_ROOT="$CG_OUTPUT_ROOT" \
  CG_WORK_ROOT="$CG_WORK_ROOT" \
  bash "$SCRIPT_DIR/run_stage4_full_sft_pipeline.sh"
  echo "[stage4-formal-18] DONE model=$model"
  date '+%F %T'
done

"$PYTHON" - "$DRIVER_MANIFEST" "$CG_OUTPUT_ROOT" "$CG_WORK_ROOT" "$EXP_NAME" "$SEED" "$MODELS_CSV" "$ARMS_CSV" <<'PY'
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
output_root = Path(sys.argv[2])
work_root = Path(sys.argv[3])
exp_name = sys.argv[4]
seed = int(sys.argv[5])
models = sys.argv[6].split(",")
arms = sys.argv[7].split(",")
runs = []
for model in models:
    for arm in arms:
        ckpt_dir = output_root / "checkpoints" / "sft_full" / exp_name / model / arm
        log_dir = work_root / "logs" / exp_name / model / arm
        result_path = log_dir / "training_result.json"
        row = {
            "model": model,
            "arm": arm,
            "checkpoint_dir": str(ckpt_dir),
            "log_dir": str(log_dir),
            "manifest_exists": (ckpt_dir / "run_manifest.json").exists(),
            "safetensor_count": len(list(ckpt_dir.glob("*.safetensors"))),
        }
        if result_path.exists():
            try:
                row.update(json.loads(result_path.read_text(encoding="utf-8")))
                row["status"] = "completed"
            except Exception as exc:
                row["status"] = "result_read_error"
                row["result_read_error"] = repr(exc)
        else:
            row["status"] = "missing_result"
        runs.append(row)
summary = {"seed": seed, "exp_name": exp_name, "models": models, "arms": arms, "runs": runs}
manifest.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
(work_root / "logs" / exp_name / "stage4_full_sft_formal_18arms_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY

echo "[stage4-formal-18] complete"
