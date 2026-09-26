#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
STAGE_DIR="$REPO_ROOT/stages/stage05_legacy_taco_dapo_ablation"
: "${CG_DATA_ROOT:?set CG_DATA_ROOT to the external dataset root}"
: "${CG_MODEL_ROOT:?set CG_MODEL_ROOT to the external model root}"
: "${CG_OUTPUT_ROOT:?set CG_OUTPUT_ROOT to the external output root}"
CG_WORK_ROOT="${CG_WORK_ROOT:-$CG_OUTPUT_ROOT/.work}"
VERL_RECIPE_DIR="${VERL_RECIPE_DIR:-${CG_VERL_RECIPE_DIR:-}}"
: "${VERL_RECIPE_DIR:?set CG_VERL_RECIPE_DIR (or VERL_RECIPE_DIR) to a compatible checkout}"
ENV_DIR=${ENV_DIR:-"${CG_VERL_ENV:-$CG_WORK_ROOT/verl_env}"}
if [[ -z "${PYTHON:-}" ]]; then
  [[ -x "$ENV_DIR/bin/python" ]] && PYTHON="$ENV_DIR/bin/python" || PYTHON=python
fi
if [[ -z "${RAY:-}" ]]; then
  [[ -x "$ENV_DIR/bin/ray" ]] && RAY="$ENV_DIR/bin/ray" || RAY=ray
fi
[[ -x "$ENV_DIR/bin/python" ]] || {
  echo "[fatal] DAPO requires an isolated environment with bin/python: $ENV_DIR" >&2
  exit 2
}
MODEL_PATH=${MODEL_PATH:-"$CG_MODEL_ROOT/Qwen3-8B-Base"}
DATA_DIR=${DATA_DIR:-"$CG_DATA_ROOT/stage3_dapo_full_verified"}
REWARD_PATH=${REWARD_PATH:-"$STAGE_DIR/stage3_code_reward.py"}
VERL_ENTRYPOINT=${VERL_ENTRYPOINT:-dapo.main_dapo}
VERL_CONFIG_CWD=${VERL_CONFIG_CWD:-"$VERL_RECIPE_DIR"}

MAX_RESPONSE_LENGTH=${1:-2048}
TOTAL_STEPS=${2:-10}
TRAIN_BATCH_SIZE=${3:-10}
ROLLOUT_N=${4:-4}
GEN_BATCH_SIZE=${GEN_BATCH_SIZE:-$((TRAIN_BATCH_SIZE * 3))}

N_GPUS=${N_GPUS:-1}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
LR=${LR:-5e-6}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}
LR_WARMUP_STEPS=${LR_WARMUP_STEPS:-0}
LORA_RANK=${LORA_RANK:-16}
LORA_ALPHA=${LORA_ALPHA:-32}
LOSS_AGG_MODE=${LOSS_AGG_MODE:-token-mean}
CLIP_RATIO_C=${CLIP_RATIO_C:-10.0}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1.0}
ROLLOUT_TOP_P=${ROLLOUT_TOP_P:-1.0}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.30}
PARAM_OFFLOAD=${PARAM_OFFLOAD:-True}
OPTIMIZER_OFFLOAD=${OPTIMIZER_OFFLOAD:-True}
USE_ORIG_PARAMS=${USE_ORIG_PARAMS:-True}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8192}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-32}
REWARD_WORKERS=${REWARD_WORKERS:-4}
USE_DYNAMIC_BSZ=${USE_DYNAMIC_BSZ:-False}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}
TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE:-1}
ROLLOUT_LOAD_FORMAT=${ROLLOUT_LOAD_FORMAT:-dummy}
LAYERED_SUMMON=${LAYERED_SUMMON:-False}
MULTI_STAGE_WAKE_UP=${MULTI_STAGE_WAKE_UP:-False}
UPDATE_WEIGHTS_BUCKET_MEGABYTES=${UPDATE_WEIGHTS_BUCKET_MEGABYTES:-2048}
RAY_NUM_CPUS=${RAY_NUM_CPUS:-}
RAY_OBJECT_STORE_MEMORY=${RAY_OBJECT_STORE_MEMORY:-}
PROJECT_NAME=${PROJECT_NAME:-stage05_legacy_taco_dapo_ablation}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_8b_lora_r${LORA_RANK}_max${MAX_RESPONSE_LENGTH}}
FILTER_GROUPS_ENABLE=${FILTER_GROUPS_ENABLE:-True}
FILTER_GROUPS_METRIC=${FILTER_GROUPS_METRIC:-seq_reward}
FILTER_GROUPS_MAX_GEN_BATCHES=${FILTER_GROUPS_MAX_GEN_BATCHES:-0}
DATA_SHUFFLE=${DATA_SHUFFLE:-True}
: "${DATA_SEED:?Set DATA_SEED explicitly for this run}"
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}
TEST_FREQ=${TEST_FREQ:-10}
SAVE_FREQ=${SAVE_FREQ:-50}
MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-2}
RESUME_MODE=${RESUME_MODE:-auto}
VAL_DO_SAMPLE=${VAL_DO_SAMPLE:-False}
VAL_ROLLOUT_N=${VAL_ROLLOUT_N:-1}
REWARD_EXEC_TIMEOUT=${REWARD_EXEC_TIMEOUT:-180.0}
REWARD_PER_TEST_TIMEOUT=${REWARD_PER_TEST_TIMEOUT:-1.0}
REWARD_MEMORY_LIMIT_GIB=${REWARD_MEMORY_LIMIT_GIB:-2.0}
REWARD_OUTPUT_LIMIT_MIB=${REWARD_OUTPUT_LIMIT_MIB:-16.0}
OVERLONG_REWARD_ENABLE=${OVERLONG_REWARD_ENABLE:-False}
OVERLONG_REWARD_BUFFER=${OVERLONG_REWARD_BUFFER:-0}
OVERLONG_REWARD_PENALTY=${OVERLONG_REWARD_PENALTY:-0.0}
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
RUN_DIR=${RUN_DIR:-"$CG_OUTPUT_ROOT/stage05_legacy_taco_dapo_ablation/qwen3_8b_lora_r${LORA_RANK}_max${MAX_RESPONSE_LENGTH}_${TIMESTAMP}"}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-"$RUN_DIR/checkpoints"}
ADAPTER_DIR=${ADAPTER_DIR:-"$RUN_DIR/adapters"}
RAY_TMP_BASE=${RAY_TMP_BASE:-"$CG_WORK_ROOT/tmp/ray_stage05_${MAX_RESPONSE_LENGTH}_${TIMESTAMP}_$$"}
LOG_DIR=${LOG_DIR:-"$CG_WORK_ROOT/logs/stage05_legacy_taco_dapo_ablation"}

mkdir -p "$RUN_DIR/rollouts" "$CHECKPOINT_DIR" "$ADAPTER_DIR" "$RAY_TMP_BASE" "$LOG_DIR"

command -v "$PYTHON" >/dev/null 2>&1 || { echo "[fatal] python command not found: $PYTHON" >&2; exit 2; }
[[ -d "$MODEL_PATH" ]] || { echo "[fatal] missing model: $MODEL_PATH" >&2; exit 2; }
[[ -d "$VERL_RECIPE_DIR" ]] || { echo "[fatal] missing verl-recipe checkout: $VERL_RECIPE_DIR" >&2; exit 2; }
[[ -f "$REWARD_PATH" ]] || { echo "[fatal] missing reward function: $REWARD_PATH" >&2; exit 2; }

PATCH_ACTION=${CG_VERL_PATCH_ACTION:-apply}
case "$PATCH_ACTION" in
  apply|check) ;;
  *) echo "[fatal] CG_VERL_PATCH_ACTION must be apply or check" >&2; exit 2 ;;
esac
PATCH_ARGS=(
  "$SCRIPT_DIR/apply_runtime_patches.py" "$PATCH_ACTION"
  --environment "$ENV_DIR"
  --recipe "$VERL_RECIPE_DIR"
)
if [[ "${CG_ALLOW_UNVERIFIED_VERL:-0}" == "1" ]]; then
  PATCH_ARGS+=(--allow-unverified-recipe)
fi
"$PYTHON" "${PATCH_ARGS[@]}" > "$RUN_DIR/runtime_patch_manifest.stdout.json"
CHECK_ARGS=(
  "$SCRIPT_DIR/apply_runtime_patches.py" check
  --environment "$ENV_DIR"
  --recipe "$VERL_RECIPE_DIR"
)
if [[ "${CG_ALLOW_UNVERIFIED_VERL:-0}" == "1" ]]; then
  CHECK_ARGS+=(--allow-unverified-recipe)
fi
"$PYTHON" "${CHECK_ARGS[@]}" > "$RUN_DIR/runtime_patch_check.stdout.json"

echo "run_dir=$RUN_DIR"
echo "max_response_length=$MAX_RESPONSE_LENGTH total_steps=$TOTAL_STEPS train_batch_size=$TRAIN_BATCH_SIZE rollout_n=$ROLLOUT_N"
echo "verl_entrypoint=$VERL_ENTRYPOINT gen_batch_size=$GEN_BATCH_SIZE filter_groups=$FILTER_GROUPS_ENABLE metric=$FILTER_GROUPS_METRIC max_gen_batches=$FILTER_GROUPS_MAX_GEN_BATCHES"
echo "n_gpus=$N_GPUS project_name=$PROJECT_NAME experiment_name=$EXPERIMENT_NAME"
echo "gpu_memory_utilization=$GPU_MEMORY_UTILIZATION param_offload=$PARAM_OFFLOAD optimizer_offload=$OPTIMIZER_OFFLOAD use_orig_params=$USE_ORIG_PARAMS"
echo "lr=$LR weight_decay=$WEIGHT_DECAY lr_warmup_steps=$LR_WARMUP_STEPS loss_agg_mode=$LOSS_AGG_MODE clip_ratio_c=$CLIP_RATIO_C"
echo "rollout_temperature=$ROLLOUT_TEMPERATURE rollout_top_p=$ROLLOUT_TOP_P"
echo "data_shuffle=$DATA_SHUFFLE data_seed=$DATA_SEED val_before_train=$VAL_BEFORE_TRAIN test_freq=$TEST_FREQ save_freq=$SAVE_FREQ"
echo "checkpoint_dir=$CHECKPOINT_DIR adapter_dir=$ADAPTER_DIR max_actor_ckpt_to_keep=$MAX_ACTOR_CKPT_TO_KEEP"
echo "max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS max_num_seqs=$MAX_NUM_SEQS reward_workers=$REWARD_WORKERS use_dynamic_bsz=$USE_DYNAMIC_BSZ"
echo "tensor_model_parallel_size=$TENSOR_MODEL_PARALLEL_SIZE rollout_load_format=$ROLLOUT_LOAD_FORMAT layered_summon=$LAYERED_SUMMON"
echo "multi_stage_wake_up=$MULTI_STAGE_WAKE_UP update_weights_bucket_megabytes=$UPDATE_WEIGHTS_BUCKET_MEGABYTES"
echo "ray_num_cpus=${RAY_NUM_CPUS:-auto} ray_object_store_memory=${RAY_OBJECT_STORE_MEMORY:-auto}"

if [[ ! -f "$DATA_DIR/train.parquet" || ! -f "$DATA_DIR/val.parquet" ]]; then
  echo "[fatal] expected train.parquet and val.parquet under $DATA_DIR" >&2
  echo "[hint] prepare them with $STAGE_DIR/prepare_stage3_dapo_full_data.py or prepare_stage3_dapo_ablation_data.py" >&2
  exit 2
fi

DEFAULT_CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((N_GPUS - 1)))
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$DEFAULT_CUDA_VISIBLE_DEVICES}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH="$ENV_DIR/bin:/usr/local/cuda/bin:${PATH:-}"
export LD_LIBRARY_PATH="$ENV_DIR/lib:/usr/local/cuda/compat:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export CPATH="/usr/local/cuda/include:${CPATH:-}"
export LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}"
export TRITON_PTXAS_PATH=${TRITON_PTXAS_PATH:-"$ENV_DIR/bin/ptxas"}
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
if [[ -n "${VLLM_ATTENTION_BACKEND:-}" ]]; then
  export VLLM_ATTENTION_BACKEND
fi
export HYDRA_FULL_ERROR=1
export RAY_TMPDIR="$RAY_TMP_BASE"
export TMPDIR="$RAY_TMP_BASE"
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1
export PYTHONPATH="$SCRIPT_DIR/sitecustomize:$VERL_RECIPE_DIR:${PYTHONPATH:-}"
export VERL_LOGGING_LEVEL=INFO

if command -v "$RAY" >/dev/null 2>&1; then
  "$RAY" stop --force >/dev/null 2>&1 || true
fi

"$PYTHON" -u - "$RUN_DIR/memory.csv" <<'PY' &
import csv
import datetime
import sys
import time

import torch

path = sys.argv[1]
device_count = torch.cuda.device_count()
header = ["timestamp"]
for index in range(device_count):
    header.extend([f"gpu{index}_used_mib", f"gpu{index}_total_mib"])
with open(path, "w", newline="", encoding="utf-8") as handle:
    csv.writer(handle).writerow(header)

while True:
    try:
        row = [datetime.datetime.now().isoformat(timespec="seconds")]
        for index in range(device_count):
            free_bytes, total_bytes = torch.cuda.mem_get_info(index)
            used_mib = (total_bytes - free_bytes) / (1024 * 1024)
            total_mib = total_bytes / (1024 * 1024)
            row.extend([f"{used_mib:.1f}", f"{total_mib:.1f}"])
        with open(path, "a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(row)
    except Exception:
        pass
    time.sleep(1)
PY
MONITOR_PID=$!

cleanup() {
  kill "$MONITOR_PID" >/dev/null 2>&1 || true
  wait "$MONITOR_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
PPO_MAX_TOKEN_LEN=${PPO_MAX_TOKEN_LEN:-$((MAX_MODEL_LEN + 512))}
LOG_PROB_MAX_TOKEN_LEN=${LOG_PROB_MAX_TOKEN_LEN:-$PPO_MAX_TOKEN_LEN}

CMD=(
  "$PYTHON" -m "$VERL_ENTRYPOINT"
  "data.train_files=$DATA_DIR/train.parquet"
  "data.val_files=$DATA_DIR/val.parquet"
  "data.prompt_key=prompt"
  "data.reward_fn_key=data_source"
  "data.train_batch_size=$TRAIN_BATCH_SIZE"
  "data.gen_batch_size=$GEN_BATCH_SIZE"
  "data.max_prompt_length=$MAX_PROMPT_LENGTH"
  "data.max_response_length=$MAX_RESPONSE_LENGTH"
  "data.truncation=error"
  "data.filter_overlong_prompts=True"
  "data.filter_overlong_prompts_workers=4"
  "data.shuffle=$DATA_SHUFFLE"
  "+data.seed=$DATA_SEED"
  "data.dataloader_num_workers=0"
  "data.trust_remote_code=True"
  "actor_rollout_ref.model.path=$MODEL_PATH"
  "actor_rollout_ref.model.trust_remote_code=True"
  "actor_rollout_ref.model.use_remove_padding=True"
  "actor_rollout_ref.model.enable_gradient_checkpointing=True"
  "+actor_rollout_ref.model.override_config.attn_implementation=sdpa"
  "actor_rollout_ref.model.lora_rank=$LORA_RANK"
  "actor_rollout_ref.model.lora_alpha=$LORA_ALPHA"
  "actor_rollout_ref.model.target_modules=all-linear"
  "actor_rollout_ref.actor.optim.lr=$LR"
  "actor_rollout_ref.actor.optim.weight_decay=$WEIGHT_DECAY"
  "actor_rollout_ref.actor.optim.lr_warmup_steps=$LR_WARMUP_STEPS"
  "actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BATCH_SIZE"
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU"
  "actor_rollout_ref.actor.use_dynamic_bsz=$USE_DYNAMIC_BSZ"
  "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN"
  "actor_rollout_ref.actor.clip_ratio=0.2"
  "actor_rollout_ref.actor.clip_ratio_low=0.2"
  "actor_rollout_ref.actor.clip_ratio_high=0.28"
  "actor_rollout_ref.actor.clip_ratio_c=$CLIP_RATIO_C"
  "actor_rollout_ref.actor.loss_agg_mode=$LOSS_AGG_MODE"
  "actor_rollout_ref.actor.use_kl_loss=False"
  "actor_rollout_ref.actor.entropy_coeff=0.0"
  "actor_rollout_ref.actor.ppo_epochs=1"
  "actor_rollout_ref.actor.shuffle=False"
  "actor_rollout_ref.actor.use_torch_compile=False"
  "actor_rollout_ref.actor.fsdp_config.param_offload=$PARAM_OFFLOAD"
  "actor_rollout_ref.actor.fsdp_config.optimizer_offload=$OPTIMIZER_OFFLOAD"
  "actor_rollout_ref.actor.fsdp_config.use_orig_params=$USE_ORIG_PARAMS"
  "actor_rollout_ref.actor.fsdp_config.use_torch_compile=False"
  "actor_rollout_ref.rollout.name=vllm"
  "actor_rollout_ref.rollout.mode=async"
  "actor_rollout_ref.rollout.n=$ROLLOUT_N"
  "actor_rollout_ref.rollout.temperature=$ROLLOUT_TEMPERATURE"
  "actor_rollout_ref.rollout.top_p=$ROLLOUT_TOP_P"
  "actor_rollout_ref.rollout.top_k=-1"
  "actor_rollout_ref.rollout.val_kwargs.do_sample=$VAL_DO_SAMPLE"
  "actor_rollout_ref.rollout.val_kwargs.n=$VAL_ROLLOUT_N"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=$TENSOR_MODEL_PARALLEL_SIZE"
  "actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION"
  "actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN"
  "actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS"
  "actor_rollout_ref.rollout.max_num_seqs=$MAX_NUM_SEQS"
  "actor_rollout_ref.rollout.load_format=$ROLLOUT_LOAD_FORMAT"
  "actor_rollout_ref.rollout.layered_summon=$LAYERED_SUMMON"
  "actor_rollout_ref.rollout.multi_stage_wake_up=$MULTI_STAGE_WAKE_UP"
  "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=$UPDATE_WEIGHTS_BUCKET_MEGABYTES"
  "actor_rollout_ref.rollout.enforce_eager=True"
  "actor_rollout_ref.rollout.free_cache_engine=True"
  "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU"
  "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=$USE_DYNAMIC_BSZ"
  "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$LOG_PROB_MAX_TOKEN_LEN"
  "algorithm.adv_estimator=grpo"
  "algorithm.norm_adv_by_std_in_grpo=True"
  "algorithm.use_kl_in_reward=False"
  "algorithm.filter_groups.enable=$FILTER_GROUPS_ENABLE"
  "algorithm.filter_groups.metric=$FILTER_GROUPS_METRIC"
  "algorithm.filter_groups.max_num_gen_batches=$FILTER_GROUPS_MAX_GEN_BATCHES"
  "reward.reward_manager.name=dapo"
  "reward.num_workers=$REWARD_WORKERS"
  "reward.custom_reward_function.path=$REWARD_PATH"
  "reward.custom_reward_function.name=compute_score"
  "+reward.custom_reward_function.reward_kwargs.exec_timeout=$REWARD_EXEC_TIMEOUT"
  "+reward.custom_reward_function.reward_kwargs.per_test_timeout=$REWARD_PER_TEST_TIMEOUT"
  "+reward.custom_reward_function.reward_kwargs.memory_limit_gib=$REWARD_MEMORY_LIMIT_GIB"
  "+reward.custom_reward_function.reward_kwargs.output_limit_mib=$REWARD_OUTPUT_LIMIT_MIB"
  "reward.reward_kwargs.max_resp_len=$MAX_RESPONSE_LENGTH"
  "reward.reward_kwargs.overlong_buffer_cfg.enable=$OVERLONG_REWARD_ENABLE"
  "reward.reward_kwargs.overlong_buffer_cfg.len=$OVERLONG_REWARD_BUFFER"
  "reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=$OVERLONG_REWARD_PENALTY"
  "reward.reward_kwargs.overlong_buffer_cfg.log=True"
  "trainer.project_name=$PROJECT_NAME"
  "trainer.experiment_name=$EXPERIMENT_NAME"
  "trainer.logger=['console']"
  "trainer.nnodes=1"
  "trainer.n_gpus_per_node=$N_GPUS"
  "trainer.total_epochs=1"
  "trainer.total_training_steps=$TOTAL_STEPS"
  "trainer.val_before_train=$VAL_BEFORE_TRAIN"
  "trainer.test_freq=$TEST_FREQ"
  "trainer.save_freq=$SAVE_FREQ"
  "trainer.resume_mode=$RESUME_MODE"
  "trainer.max_actor_ckpt_to_keep=$MAX_ACTOR_CKPT_TO_KEEP"
  "trainer.default_local_dir=$CHECKPOINT_DIR"
  "+trainer.adapter_dir=$ADAPTER_DIR"
  "+trainer.adapter_layered_summon=$LAYERED_SUMMON"
  "trainer.rollout_data_dir=$RUN_DIR/rollouts"
  "trainer.validation_data_dir=$RUN_DIR/validation"
  "+trainer.utilization_log_path=$RUN_DIR/utilization_events.jsonl"
  "+trainer.utilization_summary_path=$RUN_DIR/utilization_summary.json"
  "+ray_kwargs.ray_init._temp_dir=$RAY_TMP_BASE"
  "+ray_kwargs.ray_init.include_dashboard=False"
  "hydra.run.dir=$RUN_DIR/hydra"
)

if [[ -n "$RAY_NUM_CPUS" ]]; then
  CMD+=("+ray_kwargs.ray_init.num_cpus=$RAY_NUM_CPUS")
fi
if [[ -n "$RAY_OBJECT_STORE_MEMORY" ]]; then
  CMD+=("+ray_kwargs.ray_init.object_store_memory=$RAY_OBJECT_STORE_MEMORY")
fi
if [[ -n "${ROLLOUT_HF_OVERRIDES_ROPE_THETA:-}" ]]; then
  CMD+=("+actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides.rope_theta=$ROLLOUT_HF_OVERRIDES_ROPE_THETA")
fi
if [[ -n "${VLLM_ATTENTION_BACKEND:-}" ]]; then
  CMD+=("+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_ATTENTION_BACKEND=$VLLM_ATTENTION_BACKEND")
fi

printf '%q ' "${CMD[@]}" > "$RUN_DIR/command.txt"
printf '\n' >> "$RUN_DIR/command.txt"

set +e
(
  cd "$VERL_CONFIG_CWD"
  "${CMD[@]}"
) 2>&1 | tee "$RUN_DIR/train.log" "$LOG_DIR/$(basename "$RUN_DIR").log"
TRAIN_STATUS=${PIPESTATUS[0]}
set -e

cleanup
trap - EXIT

if [[ "$TRAIN_STATUS" -ne 0 && -d "$RAY_TMP_BASE/session_latest/logs" ]]; then
  mkdir -p "$RUN_DIR/ray_logs"
  cp -a "$RAY_TMP_BASE/session_latest/logs/." "$RUN_DIR/ray_logs/" 2>/dev/null || true
fi

if command -v "$RAY" >/dev/null 2>&1; then
  "$RAY" stop --force >/dev/null 2>&1 || true
fi

exit "$TRAIN_STATUS"
