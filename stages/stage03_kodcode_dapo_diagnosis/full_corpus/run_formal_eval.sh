#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STAGE_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd -- "$STAGE_ROOT/../.." && pwd)"
: "${CG_MODEL_ROOT:?set CG_MODEL_ROOT to the external model root}"
: "${CG_OUTPUT_ROOT:?set CG_OUTPUT_ROOT to the external output root}"
: "${CG_DATA_TEST_ROOT:?set CG_DATA_TEST_ROOT to the external data_test root}"
WORK="${WORK:-$CG_OUTPUT_ROOT/stage03_kodcode_dapo_diagnosis/full_corpus}"
PYTHON="${PYTHON:-python}"
EVALUATOR="${EVALUATOR:-$REPO_ROOT/evaluation/paper_suite.py}"
SEED="${SEED:-20260725}"
TASKS="arc_challenge,scienceqa,legalbench,gsm8k,math500_medium,math500_high_level,finqa,medcalc,humaneval,mbpp_plus,mbpp_simple"

for MODEL_KEY in qwen25; do
  case "$MODEL_KEY" in
    qwen25) MODEL_NAME="Qwen2.5-7B-Instruct"; TRAIN_PROMPT_MODE=native; PROMPT_MODE=chat ;;
  esac
  if [[ "$TRAIN_PROMPT_MODE" == "native" && "$PROMPT_MODE" != "chat" ]]; then
    echo "prompt contract mismatch for $MODEL_KEY: native training requires native chat evaluation" >&2
    exit 2
  fi
  if [[ "$TRAIN_PROMPT_MODE" == "plain" && "$PROMPT_MODE" != "plain" ]]; then
    echo "prompt contract mismatch for $MODEL_KEY: plain training requires plain evaluation" >&2
    exit 2
  fi
  BASE_MODEL="$CG_MODEL_ROOT/$MODEL_NAME"
  CKPT_ROOT="$WORK/checkpoints/$MODEL_KEY/full_dapo"
  [[ -f "$CKPT_ROOT/.training_complete" ]] || {
    echo "training incomplete for $MODEL_KEY" >&2
    exit 2
  }
  SELECTION_FILE="$WORK/results/selection/$MODEL_KEY.json"
  BEST_ADAPTER="$("$PYTHON" "$STAGE_ROOT/shared/select_best_adapter.py" \
    --checkpoint-root "$CKPT_ROOT" \
    --output-root "$WORK/output" \
    --model-key "$MODEL_KEY" \
    --selection-file "$SELECTION_FILE")"
  for STAGE in base dapo_best; do
    OUTPUT_DIR="$WORK/results/formal/$MODEL_KEY/$STAGE"
    mkdir -p "$OUTPUT_DIR"
    ARGS=(
      --model-name "${MODEL_KEY}_${STAGE}"
      --data-root "$CG_DATA_TEST_ROOT"
      --base-model-path "$BASE_MODEL"
      --output-dir "$OUTPUT_DIR"
      --seed "$SEED"
      --prompt-mode "$PROMPT_MODE"
      --temperature 0.2
      --top-p 0.95
      --max-new-tokens-code 2048
      --choice-batch-size 24
      --generation-batch-size 4
      --tasks "$TASKS"
      --limit 0
      --resume
      --skip-healthbench-generation
    )
    if [[ "$STAGE" == "dapo_best" ]]; then
      ARGS+=(--adapter-path "$BEST_ADAPTER")
    fi
    echo "[eval] model=$MODEL_KEY stage=$STAGE mode=$PROMPT_MODE"
    "$PYTHON" "$EVALUATOR" "${ARGS[@]}" 2>&1 | tee "$OUTPUT_DIR/eval.log"
  done
done

"$PYTHON" "$STAGE_ROOT/shared/summarize_results.py" --work-dir "$WORK" \
  --models qwen25
