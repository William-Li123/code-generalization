#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STAGE_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
: "${CG_MODEL_ROOT:?set CG_MODEL_ROOT to the external model root}"
: "${CG_OUTPUT_ROOT:?set CG_OUTPUT_ROOT to the external output root}"
WORK="${WORK:-$CG_OUTPUT_ROOT/stage03_kodcode_dapo_diagnosis/full_corpus}"
PYTHON="${PYTHON:-python}"
DATA_DIR="${DATA_DIR:-$WORK/data}"
STATUS="$WORK/logs/status.tsv"
READY_MARKER="$WORK/data/.data_ready"
FAILED_MARKER="$WORK/data/.data_failed"
DATA_READY_WAIT_SECONDS="${DATA_READY_WAIT_SECONDS:-21600}"
mkdir -p "$WORK/logs"
[[ -f "$STATUS" ]] || printf 'phase\tstatus\ttimestamp\tdetail\n' > "$STATUS"

log_status() {
  printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$(date -Iseconds)" "$3" | tee -a "$STATUS"
}

WAIT_START=$(date +%s)
while [[ ! -f "$READY_MARKER" ]]; do
  if [[ -f "$FAILED_MARKER" ]]; then
    echo "data preparation failed; refusing to start GPU training" >&2
    cat "$FAILED_MARKER" >&2
    exit 10
  fi
  NOW=$(date +%s)
  if (( NOW - WAIT_START >= DATA_READY_WAIT_SECONDS )); then
    echo "timed out waiting for verified data marker: $READY_MARKER" >&2
    exit 11
  fi
  echo "[wait-data] verified parquet is not ready; sleeping 60 seconds"
  sleep 60
done
"$PYTHON" "$SCRIPT_DIR/static_check.py" \
  --data-dir "$DATA_DIR" \
  --model-root "$CG_MODEL_ROOT" \
  --plain-template "$STAGE_ROOT/plain_chat_template.jinja" \
  --reward-path "$STAGE_ROOT/shared/kodcode_reward.py"

log_status train start "Qwen2.5-7B-Instruct full eligible 37,881-source DAPO"
if WORK="$WORK" DATA_DIR="$DATA_DIR" bash "$SCRIPT_DIR/run_qwen25.sh"; then
  log_status train complete "final adapter validated"
else
  code=$?
  log_status train failed "exit=$code"
  exit "$code"
fi

log_status eval start "matching native-chat base and best-validation adapter"
if WORK="$WORK" bash "$SCRIPT_DIR/run_formal_eval.sh"; then
  log_status eval complete "2 groups x 11 tasks"
else
  code=$?
  log_status eval failed "exit=$code"
  exit "$code"
fi
