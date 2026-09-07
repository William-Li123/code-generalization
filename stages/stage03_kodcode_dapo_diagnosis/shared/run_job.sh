#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STAGE_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
DERIVED_REPO_ROOT="$(cd -- "$STAGE_ROOT/../.." && pwd)"
REPO_ROOT="${REPO_ROOT:-$DERIVED_REPO_ROOT}"
: "${CG_MODEL_ROOT:?set CG_MODEL_ROOT to the external model root}"
: "${CG_OUTPUT_ROOT:?set CG_OUTPUT_ROOT to the external output root}"
: "${CG_WORK_ROOT:?set CG_WORK_ROOT to the external scratch/work root}"

SOURCE_MODEL_PATH="${SOURCE_MODEL_PATH:-$CG_MODEL_ROOT/Qwen3-8B-Base}"
MODEL_NAME="${MODEL_NAME:-$(basename "$SOURCE_MODEL_PATH")}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"
TOTAL_STEPS="${TOTAL_STEPS:-5}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-20}"
ROLLOUT_N="${ROLLOUT_N:-8}"
TEST_ID="${TEST_ID:-$(date +%Y%m%d-%H%M%S)}"
RUN_NAME="${RUN_NAME:-test-stage3-dapo-4x80-max${MAX_RESPONSE_LENGTH}-${TEST_ID}}"
RUN_DIR="${RUN_DIR:-$CG_OUTPUT_ROOT/stage03_kodcode_dapo_diagnosis/runs/$RUN_NAME}"
SYSTEM_LOG="${SYSTEM_LOG:-$CG_OUTPUT_ROOT/stage03_kodcode_dapo_diagnosis/logs/$RUN_NAME.system.log}"

# The archived run used a disposable, patched Verl environment. Prefer an
# already isolated CG_VERL_ENV; CG_VERL_ENV_ARCHIVE retains the archive-based
# workflow without assuming a private runtime_cache location.
ENV_ARCHIVE="${CG_VERL_ENV_ARCHIVE:-${ENV_ARCHIVE:-}}"
VERL_ENV_SOURCE="${CG_VERL_ENV:-${ENV_DIR:-}}"
LOCAL_RUNTIME_ROOT="${LOCAL_RUNTIME_ROOT:-$CG_WORK_ROOT/stage03_kodcode_dapo_diagnosis/runtime}"
if [[ -n "$ENV_ARCHIVE" ]]; then
  [[ -f "$ENV_ARCHIVE" ]] || { echo "missing Verl environment archive: $ENV_ARCHIVE" >&2; exit 3; }
  LOCAL_ENV_ARCHIVE="$LOCAL_RUNTIME_ROOT/verl_env.tar"
  LOCAL_ENV_DIR="${LOCAL_ENV_DIR:-$LOCAL_RUNTIME_ROOT/verl}"
  if [[ ! -x "$LOCAL_ENV_DIR/bin/python" ]]; then
    rm -rf "$LOCAL_RUNTIME_ROOT"
    mkdir -p "$LOCAL_RUNTIME_ROOT"
    cp "$ENV_ARCHIVE" "$LOCAL_ENV_ARCHIVE"
    tar -xf "$LOCAL_ENV_ARCHIVE" -C "$LOCAL_RUNTIME_ROOT"
    rm -f "$LOCAL_ENV_ARCHIVE"
  fi
elif [[ -n "$VERL_ENV_SOURCE" ]]; then
  LOCAL_ENV_DIR="${LOCAL_ENV_DIR:-$VERL_ENV_SOURCE}"
else
  echo "set CG_VERL_ENV to an isolated Verl environment or CG_VERL_ENV_ARCHIVE to its tar archive" >&2
  exit 3
fi
[[ -x "$LOCAL_ENV_DIR/bin/python" ]] || {
  echo "missing Verl Python: $LOCAL_ENV_DIR/bin/python" >&2
  exit 3
}
VERL_RECIPE_DIR="${VERL_RECIPE_DIR:-${CG_VERL_RECIPE_DIR:-}}"
: "${VERL_RECIPE_DIR:?set CG_VERL_RECIPE_DIR or VERL_RECIPE_DIR to the frozen Verl recipe checkout}"

mkdir -p "$RUN_DIR" "$(dirname "$SYSTEM_LOG")"
cd "$REPO_ROOT"

if [[ -f "$RUN_DIR/train.log" || -f "$RUN_DIR/memory.csv" || -f "$SYSTEM_LOG" ]]; then
  ATTEMPT_ARCHIVE="$RUN_DIR/attempts/$(date +%Y%m%d-%H%M%S)"
  mkdir -p "$ATTEMPT_ARCHIVE"
  for path in \
    "$RUN_DIR/train.log" \
    "$RUN_DIR/memory.csv" \
    "$RUN_DIR/command.txt" \
    "$RUN_DIR/utilization_events.jsonl" \
    "$RUN_DIR/utilization_summary.json" \
    "$RUN_DIR/analyze.log" \
    "$SYSTEM_LOG"; do
    if [[ -f "$path" ]]; then
      mv "$path" "$ATTEMPT_ARCHIVE/"
    fi
  done
fi

{
  echo "run_name=$RUN_NAME"
  echo "run_dir=$RUN_DIR"
  echo "source_model_path=$SOURCE_MODEL_PATH"
  echo "repo_root=$REPO_ROOT"
  echo "hostname=$(hostname)"
  date '+%F %T %Z'
  df -h /dev/shm "$CG_WORK_ROOT" "$CG_OUTPUT_ROOT" || true
  ulimit -a || true
  nvidia-smi
  "$LOCAL_ENV_DIR/bin/python" - <<'PY'
import torch
print(f"torch={torch.__version__}")
print(f"cuda={torch.version.cuda}")
print(f"gpu_count={torch.cuda.device_count()}")
for index in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(index)
    print(f"gpu{index}={props.name} total_gib={props.total_memory / 1024**3:.2f}")
PY
} | tee "$SYSTEM_LOG"

GPU_COUNT=$(nvidia-smi -L | wc -l | tr -d ' ')
EXPECTED_GPUS="${EXPECTED_GPUS:-4}"
if [[ "$GPU_COUNT" != "$EXPECTED_GPUS" ]]; then
  echo "Expected $EXPECTED_GPUS GPUs, found $GPU_COUNT" | tee -a "$SYSTEM_LOG"
  exit 2
fi

SITE_PACKAGES="$("$LOCAL_ENV_DIR/bin/python" - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"

LOCAL_MODEL_ROOT="${LOCAL_MODEL_ROOT:-$CG_WORK_ROOT/stage03_kodcode_dapo_diagnosis/model_cache/default}"
LOCAL_MODEL_PATH="$LOCAL_MODEL_ROOT/$MODEL_NAME"
if [[ ! -f "$LOCAL_MODEL_PATH/config.json" ]]; then
  mkdir -p "$LOCAL_MODEL_ROOT"
  echo "local_model_copy_start=$(date -Iseconds)" | tee -a "$SYSTEM_LOG"
  cp -a "$SOURCE_MODEL_PATH" "$LOCAL_MODEL_PATH"
  echo "local_model_copy_end=$(date -Iseconds)" | tee -a "$SYSTEM_LOG"
fi

# Qwen3 Base is intentionally trained with a raw/plain prompt contract. Patch
# only the disposable local model copy; native runs leave the tokenizer intact.
if [[ -n "${TOKENIZER_CHAT_TEMPLATE_FILE:-}" ]]; then
  "$LOCAL_ENV_DIR/bin/python" - "$LOCAL_MODEL_PATH/tokenizer_config.json" "$TOKENIZER_CHAT_TEMPLATE_FILE" <<'PY'
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
template_path = Path(sys.argv[2])
config = json.loads(config_path.read_text(encoding="utf-8"))
config["chat_template"] = template_path.read_text(encoding="utf-8")
config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"local_tokenizer_template={template_path}")
PY
fi

"$LOCAL_ENV_DIR/bin/python" - <<'PY' | tee -a "$SYSTEM_LOG"
import ray
import ray.dashboard.agent
import ray._private.runtime_env.agent.main
print("ray_agent_import_prewarm=ok")
PY

: "${DATA_DIR:?set DATA_DIR to the prepared train/val parquet directory}"
: "${REWARD_PATH:?set REWARD_PATH to the stage reward module}"

# Compatibility variables consumed by the shared repository runner. They all
# resolve to the clean checkout or explicitly configured external roots.
export ROOT="$REPO_ROOT"
export ENV_DIR="$LOCAL_ENV_DIR"
export PYTHON="$LOCAL_ENV_DIR/bin/python"
export RAY="${RAY:-$LOCAL_ENV_DIR/bin/ray}"
export RUN_DIR
export MODEL_PATH="$LOCAL_MODEL_PATH"
export N_GPUS="${N_GPUS:-$EXPECTED_GPUS}"
DEFAULT_CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((N_GPUS - 1)))
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$DEFAULT_CUDA_VISIBLE_DEVICES}"
export PROJECT_NAME="${PROJECT_NAME:-stage3_dapo_test}"
export EXPERIMENT_NAME="$RUN_NAME"
export DATA_DIR
export REWARD_PATH
export VERL_RECIPE_DIR
export VERL_ENTRYPOINT="${VERL_ENTRYPOINT:-dapo.main_dapo}"
export VERL_CONFIG_CWD="${VERL_CONFIG_CWD:-$SITE_PACKAGES}"
export LOG_DIR="${LOG_DIR:-$(dirname "$SYSTEM_LOG")}"
export RAY_TMP_BASE="${RAY_TMP_BASE:-$CG_WORK_ROOT/stage03_kodcode_dapo_diagnosis/ray/$RUN_NAME}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.50}"
export PARAM_OFFLOAD="${PARAM_OFFLOAD:-False}"
export OPTIMIZER_OFFLOAD="${OPTIMIZER_OFFLOAD:-False}"
export USE_ORIG_PARAMS="${USE_ORIG_PARAMS:-True}"
export USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-True}"
export TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-1}"
export ROLLOUT_LOAD_FORMAT="${ROLLOUT_LOAD_FORMAT:-safetensors}"
export LAYERED_SUMMON="${LAYERED_SUMMON:-False}"
export MULTI_STAGE_WAKE_UP="${MULTI_STAGE_WAKE_UP:-False}"
export UPDATE_WEIGHTS_BUCKET_MEGABYTES="${UPDATE_WEIGHTS_BUCKET_MEGABYTES:-2048}"
export PPO_MAX_TOKEN_LEN="${PPO_MAX_TOKEN_LEN:-8192}"
export LOG_PROB_MAX_TOKEN_LEN="${LOG_PROB_MAX_TOKEN_LEN:-16384}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
export GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-$((TRAIN_BATCH_SIZE * 3))}"
export FILTER_GROUPS_ENABLE="${FILTER_GROUPS_ENABLE:-True}"
export FILTER_GROUPS_METRIC="${FILTER_GROUPS_METRIC:-seq_reward}"
export FILTER_GROUPS_MAX_GEN_BATCHES="${FILTER_GROUPS_MAX_GEN_BATCHES:-0}"
export OVERLONG_REWARD_ENABLE="${OVERLONG_REWARD_ENABLE:-False}"
export OVERLONG_REWARD_BUFFER="${OVERLONG_REWARD_BUFFER:-0}"
export OVERLONG_REWARD_PENALTY="${OVERLONG_REWARD_PENALTY:-0.0}"
export REWARD_WORKERS="${REWARD_WORKERS:-64}"
export RAY_NUM_CPUS="${RAY_NUM_CPUS:-80}"
export RAY_OBJECT_STORE_MEMORY="${RAY_OBJECT_STORE_MEMORY:-68719476736}"
export RAY_USE_MULTIPROCESSING_CPU_COUNT=1
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export RAY_agent_register_timeout_ms="${RAY_AGENT_REGISTER_TIMEOUT_MS:-120000}"
export RAY_NUM_PRESTART_PYTHON_WORKERS="${RAY_NUM_PRESTART_PYTHON_WORKERS:-0}"

bash "$REPO_ROOT/dapo/run_training.sh" \
  "$MAX_RESPONSE_LENGTH" \
  "$TOTAL_STEPS" \
  "$TRAIN_BATCH_SIZE" \
  "$ROLLOUT_N" 2>&1 | tee -a "$SYSTEM_LOG"
