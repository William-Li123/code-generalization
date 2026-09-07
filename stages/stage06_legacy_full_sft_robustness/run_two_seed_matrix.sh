#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
: "${CG_DATA_ROOT:?set CG_DATA_ROOT to the external dataset root}"
: "${CG_MODEL_ROOT:?set CG_MODEL_ROOT to the external model root}"
: "${CG_OUTPUT_ROOT:?set CG_OUTPUT_ROOT to the external output root}"
CG_WORK_ROOT="${CG_WORK_ROOT:-$CG_OUTPUT_ROOT/.work}"
PYTHON="${PYTHON:-python}"
SEEDS_TEXT="${SEEDS:-20260603 20260604}"
VALIDATE_AFTER_TRAIN="${VALIDATE_AFTER_TRAIN:-1}"
MATRIX="${MATRIX:-$CG_WORK_ROOT/stage06_36_jobs.json}"

command -v "$PYTHON" >/dev/null 2>&1 || { echo "[fatal] python command not found: $PYTHON" >&2; exit 2; }

read -r -a SEEDS <<< "$SEEDS_TEXT"
if [[ "${SEEDS[*]}" != "20260603 20260604" ]]; then
  echo "[fatal] archived Stage 06 training seeds must be: 20260603 20260604" >&2
  exit 2
fi

mkdir -p "$(dirname "$MATRIX")"
"$PYTHON" "$SCRIPT_DIR/build_job_matrix.py" \
  --data-root "$CG_DATA_ROOT" \
  --model-root "$CG_MODEL_ROOT" \
  --output-root "$CG_OUTPUT_ROOT" \
  --work-root "$CG_WORK_ROOT" \
  --output "$MATRIX"
echo "[stage06] matrix=$MATRIX"

for seed in "${SEEDS[@]}"; do
  exp_name="stage4_taco_full_sft_seed_${seed}"
  echo "[stage06] START training_seed=$seed exp_name=$exp_name"
  SEED="$seed" EXP_NAME="$exp_name" \
    CG_DATA_ROOT="$CG_DATA_ROOT" \
    CG_MODEL_ROOT="$CG_MODEL_ROOT" \
    CG_OUTPUT_ROOT="$CG_OUTPUT_ROOT" \
    CG_WORK_ROOT="$CG_WORK_ROOT" \
    bash "$SCRIPT_DIR/run_stage4_full_sft_formal_18arms.sh"
  if [[ "$VALIDATE_AFTER_TRAIN" == "1" || "$VALIDATE_AFTER_TRAIN" == "true" ]]; then
    "$PYTHON" "$SCRIPT_DIR/validate_full_checkpoints.py" \
      --matrix "$MATRIX" --training-seed "$seed" --load
  fi
  echo "[stage06] DONE training_seed=$seed"
done

echo "[stage06] complete: 36 full checkpoints"
