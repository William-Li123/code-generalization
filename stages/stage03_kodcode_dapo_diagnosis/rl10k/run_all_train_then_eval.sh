#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${CG_MODEL_ROOT:?set CG_MODEL_ROOT to the external model root}"
: "${CG_OUTPUT_ROOT:?set CG_OUTPUT_ROOT to the external output root}"
WORK="${WORK:-$CG_OUTPUT_ROOT/stage03_kodcode_dapo_diagnosis/rl10k}"
PYTHON="${PYTHON:-python}"
DATA_DIR="${DATA_DIR:-$WORK/data}"
READY_MARKER="$DATA_DIR/.data_ready"
[[ -f "$READY_MARKER" ]] || {
  echo "verified data marker missing: $READY_MARKER" >&2
  echo "run $SCRIPT_DIR/prepare_rl10k_data.sh first" >&2
  exit 10
}
"$PYTHON" "$SCRIPT_DIR/static_check.py" \
  --data-dir "$DATA_DIR" \
  --model-root "$CG_MODEL_ROOT"
mkdir -p "$WORK/logs"
STATUS="$WORK/logs/status.tsv"
[[ -f "$STATUS" ]] || printf 'phase\tmodel\tstatus\ttimestamp\tdetail\n' > "$STATUS"

log_status() {
  printf '%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$(date -Iseconds)" "$4" | tee -a "$STATUS"
}

for MODEL_KEY in qwen25 qwen3 llama31 gemma2; do
  log_status train "$MODEL_KEY" start "serial two-GPU DAPO"
  if MODEL_KEY="$MODEL_KEY" WORK="$WORK" bash "$SCRIPT_DIR/run_one_model.sh"; then
    log_status train "$MODEL_KEY" complete "adapter validated"
  else
    code=$?
    log_status train "$MODEL_KEY" failed "exit=$code"
    exit "$code"
  fi
done

log_status eval all start "matching-mode base and best-validation adapters"
if WORK="$WORK" bash "$SCRIPT_DIR/run_formal_eval.sh"; then
  log_status eval all complete "8 groups x 11 tasks"
else
  code=$?
  log_status eval all failed "exit=$code"
  exit "$code"
fi
