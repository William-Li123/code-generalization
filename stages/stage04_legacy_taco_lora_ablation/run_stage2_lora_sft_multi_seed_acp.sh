#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
: "${CG_DATA_ROOT:?set CG_DATA_ROOT to the external dataset root}"
: "${CG_MODEL_ROOT:?set CG_MODEL_ROOT to the external model root}"
: "${CG_OUTPUT_ROOT:?set CG_OUTPUT_ROOT to the external output root}"
CG_WORK_ROOT="${CG_WORK_ROOT:-$CG_OUTPUT_ROOT/.work}"
EXP_NAME="${EXP_NAME:-stage2_taco_sft_clean_prompt}"
PYTHON="${PYTHON:-python}"
SEEDS_TEXT="${SEEDS:-20260603 20260604 20260605}"
MODELS_TEXT="${MODELS:-Qwen2.5-7B-Instruct Qwen3-8B-Base}"
DATA_DIR="${DATA_DIR:-$CG_DATA_ROOT/$EXP_NAME}"
BASE_CKPT_ROOT="${BASE_CKPT_ROOT:-$CG_OUTPUT_ROOT/checkpoints/sft_lora/$EXP_NAME}"
BASE_LOG_ROOT="${BASE_LOG_ROOT:-$CG_WORK_ROOT/logs/$EXP_NAME}"
OVERWRITE="${OVERWRITE:-0}"

MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
EPOCHS="${EPOCHS:-1.0}"
LR="${LR:-1e-4}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
ONLY_ARMS="${ONLY_ARMS:-}"

cd "$REPO_ROOT"

read -r -a SEED_LIST <<< "$SEEDS_TEXT"
read -r -a MODEL_LIST <<< "$MODELS_TEXT"

echo "[stage2-multiseed-sft] repo_root=$REPO_ROOT"
echo "[stage2-multiseed-sft] exp_name=$EXP_NAME"
echo "[stage2-multiseed-sft] seeds=${SEED_LIST[*]}"
echo "[stage2-multiseed-sft] models=${MODEL_LIST[*]}"
echo "[stage2-multiseed-sft] data_dir=$DATA_DIR"
echo "[stage2-multiseed-sft] base_ckpt_root=$BASE_CKPT_ROOT"
echo "[stage2-multiseed-sft] base_log_root=$BASE_LOG_ROOT"
echo "[stage2-multiseed-sft] overwrite=$OVERWRITE"
echo "[stage2-multiseed-sft] hostname=$(hostname)"
nvidia-smi || true

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "[fatal] python command not found: $PYTHON" >&2
  exit 2
fi
if [[ ! -d "$DATA_DIR" ]]; then
  echo "[fatal] data dir missing: $DATA_DIR" >&2
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

for arm in "${ARMS[@]}"; do
  if [[ ! -s "$DATA_DIR/${arm}.jsonl" ]]; then
    echo "[fatal] train file missing or empty: $DATA_DIR/${arm}.jsonl" >&2
    exit 2
  fi
done

for model in "${MODEL_LIST[@]}"; do
  if [[ ! -d "$CG_MODEL_ROOT/$model" ]]; then
    echo "[fatal] model path missing: $CG_MODEL_ROOT/$model" >&2
    exit 2
  fi
done

README="$BASE_CKPT_ROOT/README.md"
mkdir -p "$BASE_CKPT_ROOT" "$BASE_LOG_ROOT"
cat > "$README" <<'EOF'
# Stage2 LoRA SFT Checkpoint Layout

The original direct model folders under this directory are historical one-seed outputs.
New multi-seed LoRA SFT outputs are stored explicitly by seed:

```text
stage2_taco_sft_clean_prompt/
  seed_<seed>/
    Qwen2.5-7B-Instruct/<arm>/adapter_model.safetensors
    Qwen3-8B-Base/<arm>/adapter_model.safetensors
```

Only LoRA adapters are trained; base model weights are frozen.
EOF

submitted_manifest="$BASE_LOG_ROOT/multiseed_sft_manifest_$(date +%Y%m%d-%H%M%S).jsonl"
echo "[stage2-multiseed-sft] manifest=$submitted_manifest"

for seed in "${SEED_LIST[@]}"; do
  for model in "${MODEL_LIST[@]}"; do
    ckpt_root="$BASE_CKPT_ROOT/seed_${seed}/${model}"
    log_root="$BASE_LOG_ROOT/seed_${seed}/${model}"
    mkdir -p "$ckpt_root" "$log_root"
    for arm in "${ARMS[@]}"; do
      printf '{"seed":%s,"model":"%s","arm":"%s","adapter_path":"%s","log_dir":"%s"}\n' \
        "$seed" "$model" "$arm" "$ckpt_root/$arm" "$log_root/$arm" >> "$submitted_manifest"
    done

    echo "[stage2-multiseed-sft] START seed=$seed model=$model"
    date '+%F %T'
    SEED="$seed" \
    MODEL_NAME="$model" \
    MODEL_PATH="$CG_MODEL_ROOT/$model" \
    DATA_DIR="$DATA_DIR" \
    MODEL_CKPT_ROOT="$ckpt_root" \
    MODEL_LOG_ROOT="$log_root" \
    OVERWRITE="$OVERWRITE" \
    MAX_SEQ_LEN="$MAX_SEQ_LEN" \
    PER_DEVICE_BATCH="$PER_DEVICE_BATCH" \
    GRAD_ACCUM="$GRAD_ACCUM" \
    EPOCHS="$EPOCHS" \
    LR="$LR" \
    LORA_R="$LORA_R" \
    LORA_ALPHA="$LORA_ALPHA" \
    LORA_DROPOUT="$LORA_DROPOUT" \
    LOGGING_STEPS="$LOGGING_STEPS" \
    ONLY_ARMS="$ONLY_ARMS" \
    CG_DATA_ROOT="$CG_DATA_ROOT" \
    CG_MODEL_ROOT="$CG_MODEL_ROOT" \
    CG_OUTPUT_ROOT="$CG_OUTPUT_ROOT" \
    CG_WORK_ROOT="$CG_WORK_ROOT" \
      bash "$SCRIPT_DIR/run_stage2_qwen25_instruct_sft_pipeline.sh"
    echo "[stage2-multiseed-sft] DONE seed=$seed model=$model"
    date '+%F %T'
  done
done

"$PYTHON" - "$submitted_manifest" "$BASE_CKPT_ROOT" "$BASE_LOG_ROOT" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
ckpt_root = Path(sys.argv[2])
log_root = Path(sys.argv[3])
rows = []
for line in manifest_path.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    adapter_dir = Path(row["adapter_path"])
    adapter_file = adapter_dir / "adapter_model.safetensors"
    row["adapter_exists"] = adapter_file.exists()
    row["adapter_file"] = str(adapter_file)
    rows.append(row)
if not rows:
    raise SystemExit("empty submitted training manifest")
seeds = sorted({int(row["seed"]) for row in rows})
models = sorted({row["model"] for row in rows})
arms = list(dict.fromkeys(row["arm"] for row in rows))
summary = {
    "checkpoint_root": str(ckpt_root),
    "log_root": str(log_root),
    "submitted_manifest": str(manifest_path),
    "seeds": seeds,
    "models": models,
    "arms": arms,
    "matrix_formula": f"{len(models)} models x {len(arms)} arms x {len(seeds)} seeds",
    "expected": len(rows),
    "present": sum(row["adapter_exists"] for row in rows),
    "rows": rows,
}
out = log_root / "multiseed_sft_summary.json"
out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"expected": summary["expected"], "present": summary["present"], "summary": str(out)}, ensure_ascii=False))
if summary["present"] != summary["expected"]:
    raise SystemExit(4)
PY

echo "[stage2-multiseed-sft] complete"
