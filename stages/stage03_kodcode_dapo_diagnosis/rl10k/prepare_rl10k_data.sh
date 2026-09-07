#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STAGE_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
: "${CG_DATA_ROOT:?set CG_DATA_ROOT to the external dataset root}"
: "${CG_MODEL_ROOT:?set CG_MODEL_ROOT to the external model root}"
: "${CG_OUTPUT_ROOT:?set CG_OUTPUT_ROOT to the external output root}"
: "${CG_DATA_TEST_ROOT:?set CG_DATA_TEST_ROOT to the external data_test root}"
WORK="${WORK:-$CG_OUTPUT_ROOT/stage03_kodcode_dapo_diagnosis/rl10k}"
PYTHON="${PYTHON:-python}"
SOURCE="${SOURCE:-$CG_DATA_ROOT/kodcode4o_r1_clean38k/train.parquet}"
FINAL_PREFLIGHT="$WORK/data/audit/exact_reward_final.json"
CUMULATIVE_REJECTS="$WORK/data/audit/exact_reward_rejects_cumulative.json"
ACTIVE_MARKER="$WORK/data/.preparation_active"
READY_MARKER="$WORK/data/.data_ready"
FAILED_MARKER="$WORK/data/.data_failed"
START_ROUND="${START_ROUND:-1}"
MAX_ROUNDS="${MAX_ROUNDS:-20}"
PREFLIGHT_WORKERS="${PREFLIGHT_WORKERS:-16}"
PREFLIGHT_EXEC_TIMEOUT="${PREFLIGHT_EXEC_TIMEOUT:-30.0}"
PREFLIGHT_PER_TEST_TIMEOUT="${PREFLIGHT_PER_TEST_TIMEOUT:-2.0}"

mkdir -p "$WORK/data/audit" "$WORK/logs"
rm -f "$READY_MARKER" "$FAILED_MARKER"
printf 'started_at=%s\npid=%s\n' "$(date -Iseconds)" "$$" > "$ACTIVE_MARKER"

finish() {
  code=$?
  trap - EXIT
  rm -f "$ACTIVE_MARKER"
  if [[ "$code" != "0" ]]; then
    printf 'failed_at=%s\nexit_code=%s\n' "$(date -Iseconds)" "$code" > "$FAILED_MARKER"
  fi
  exit "$code"
}
trap finish EXIT

if [[ "$START_ROUND" == "1" || ! -f "$CUMULATIVE_REJECTS" ]]; then
  printf '{"failure_ids":[]}\n' > "$CUMULATIVE_REJECTS"
fi

READY=0
for ROUND in $(seq "$START_ROUND" "$MAX_ROUNDS"); do
  ROUND_RESULT="$WORK/data/audit/exact_reward_round_${ROUND}.json"
  "$PYTHON" "$SCRIPT_DIR/prepare_data.py" \
    --source "$SOURCE" \
    --output-dir "$WORK/data" \
    --model-root "$CG_MODEL_ROOT" \
    --data-test-root "$CG_DATA_TEST_ROOT" \
    --plain-template "$STAGE_ROOT/plain_chat_template.jinja" \
    --reward-preflight-rejects "$CUMULATIVE_REJECTS" \
    2>&1 | tee "$WORK/logs/prepare_round_${ROUND}.log"

  "$PYTHON" "$SCRIPT_DIR/preflight_reward_dataset.py" \
    --source "$SOURCE" \
    --data-dir "$WORK/data" \
    --reward-path "$STAGE_ROOT/shared/kodcode_reward.py" \
    --output "$ROUND_RESULT" \
    --workers "$PREFLIGHT_WORKERS" \
    --exec-timeout "$PREFLIGHT_EXEC_TIMEOUT" \
    --per-test-timeout "$PREFLIGHT_PER_TEST_TIMEOUT" \
    --allow-failures \
    2>&1 | tee "$WORK/logs/reward_preflight_round_${ROUND}.log"

  FAILED="$("$PYTHON" - "$ROUND_RESULT" <<'PY'
import json
import sys
print(int(json.load(open(sys.argv[1], encoding="utf-8"))["failed"]))
PY
)"
  if [[ "$FAILED" == "0" ]]; then
    cp "$ROUND_RESULT" "$FINAL_PREFLIGHT"
    cp "${ROUND_RESULT%.json}.csv" "${FINAL_PREFLIGHT%.json}.csv"
    READY=1
    break
  fi
  "$PYTHON" - "$CUMULATIVE_REJECTS" "$ROUND_RESULT" <<'PY'
import json
import sys
from pathlib import Path

cumulative_path = Path(sys.argv[1])
round_path = Path(sys.argv[2])
cumulative = set(json.loads(cumulative_path.read_text(encoding="utf-8"))["failure_ids"])
cumulative.update(json.loads(round_path.read_text(encoding="utf-8"))["failure_ids"])
cumulative_path.write_text(
    json.dumps({"failure_ids": sorted(cumulative)}, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(f"[data] cumulative exact-reward rejects={len(cumulative)}")
PY
done

if [[ "$READY" != "1" ]]; then
  echo "exact reward preflight did not converge by round $MAX_ROUNDS" >&2
  exit 4
fi

"$PYTHON" "$SCRIPT_DIR/static_check.py" \
  --data-dir "$WORK/data" \
  --model-root "$CG_MODEL_ROOT" \
  --plain-template "$STAGE_ROOT/plain_chat_template.jinja" \
  --reward-path "$STAGE_ROOT/shared/kodcode_reward.py" \
  2>&1 | tee "$WORK/logs/static_check.log"
printf 'ready_at=%s\n' "$(date -Iseconds)" > "$READY_MARKER.tmp"
"$PYTHON" - "$WORK/data/manifest.json" >> "$READY_MARKER.tmp" <<'PY'
import hashlib
import sys
from pathlib import Path

path = Path(sys.argv[1])
print(f"manifest_sha256={hashlib.sha256(path.read_bytes()).hexdigest()}")
PY
mv "$READY_MARKER.tmp" "$READY_MARKER"
echo "[data] RL10K DAPO data is ready"
