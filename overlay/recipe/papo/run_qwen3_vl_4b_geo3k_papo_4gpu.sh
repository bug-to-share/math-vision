#!/usr/bin/env bash
# Qwen3-VL-4B on Geo3K with DAPO and optional image-mask loss.
# Append --cfg job --resolve for a CPU-only config preview.
set -euo pipefail
PAPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "${PAPO_ROOT}"
PAPO_PYTHON=${PAPO_PYTHON:-python}
# Keep this experiment's Ray runtime separate from jobs on the other GPUs.
export RAY_ADDRESS=local

# ---- user-adjustable: model / data / resources --------------------------
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export NCCL_CUMEM_HOST_ENABLE=${NCCL_CUMEM_HOST_ENABLE:-0}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0}
export NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE:-0}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-INFO}

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-VL-4B-Instruct}
DATA_DIR=${DATA_DIR:-${PAPO_ROOT}/data/geo3k}
TRAIN_FILE=${TRAIN_FILE:-${DATA_DIR}/train.parquet}
EVAL_DATA_DIR=${EVAL_DATA_DIR:-${PAPO_ROOT}/data/papo-eval}
if [[ -z "${VAL_FILES:-}" ]]; then
    if [[ -n "${VAL_FILE:-}" ]]; then
        VAL_FILES=${VAL_FILE}
    else
        VAL_FILES="[${DATA_DIR}/test.parquet,${EVAL_DATA_DIR}/geoqa-test.parquet,${EVAL_DATA_DIR}/mathvista-testmini.parquet,${EVAL_DATA_DIR}/mathvision-testmini.parquet]"
    fi
fi
NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}

# ---- user-adjustable: batches / optimization ----------------------------
# Train/minibatch sizes count retained prompts: 1024 responses, two updates/step.
# Each generation round samples 128 prompts; repeat until 128 eligible groups remain.
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
GEN_BATCH_SIZE=${GEN_BATCH_SIZE:-128}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-64}
ROLLOUT_N=${ROLLOUT_N:-8}
MAX_NUM_GEN_BATCHES=${MAX_NUM_GEN_BATCHES:-16}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}
# Two live teacher-forced views: split updates into smaller microbatches.
# Retained prompts, mini-batch size, rollout count and optimizer updates are unchanged.
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-6144}
LOG_PROB_MAX_TOKEN_LEN_PER_GPU=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-12288}
ACTOR_LR=${ACTOR_LR:-1e-6}
CLIP_RATIO_LOW=${CLIP_RATIO_LOW:-0.20}
CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH:-0.28}
OVERLONG_BUFFER_LEN=${OVERLONG_BUFFER_LEN:-512}
OVERLONG_PENALTY_FACTOR=${OVERLONG_PENALTY_FACTOR:-1.0}

# ---- user-adjustable: inference ----------------------------------------
# Training changes only temperature; no truncation of the sampling distribution.
TEMPERATURE=${TEMPERATURE:-0.7}
VAL_TEMPERATURE=${VAL_TEMPERATURE:-0.7}
VAL_TOP_P=${VAL_TOP_P:-0.8}
VAL_TOP_K=${VAL_TOP_K:-20}
VAL_PRESENCE_PENALTY=${VAL_PRESENCE_PENALTY:-1.5}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.75}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-64}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-4096}
MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))

# ---- user-adjustable: run control --------------------------------------
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-120}
# Dynamic sampling consumes several data batches per update; epochs are a ceiling.
# Geo3K has 16 generation batches/epoch, enough for 120 steps at <=16 rounds/step.
TOTAL_EPOCHS=${TOTAL_EPOCHS:-120}
TEST_FREQ=${TEST_FREQ:-10}
SAVE_FREQ=${SAVE_FREQ:-10}
PROJECT_NAME=${PROJECT_NAME:-math_vision}
PAPO_ENABLED=${PAPO_ENABLED:-true}
PAPO_GRADIENT_MODE=${PAPO_GRADIENT_MODE:-recompute_grad}
PAPO_MASK_PROBABILITY=${PAPO_MASK_PROBABILITY:-0.6}
PAPO_PERCEPTION_COEF=${PAPO_PERCEPTION_COEF:-0.01}
PAPO_CLEAN_ENTROPY_COEF=${PAPO_CLEAN_ENTROPY_COEF:-0.03}
PAPO_MASKED_ENTROPY_COEF=${PAPO_MASKED_ENTROPY_COEF:-0.03}
RUN_TAG=${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_vl_4b_papo_d_acc_n${ROLLOUT_N}_bs${TRAIN_BATCH_SIZE}_mb${PPO_MINI_BATCH_SIZE}_len${MAX_RESPONSE_LENGTH}_steps${TOTAL_TRAINING_STEPS}_${PAPO_GRADIENT_MODE}_${RUN_TAG}}
ARTIFACT_DIR=${ARTIFACT_DIR:-${PAPO_ROOT}/outputs/${PROJECT_NAME}/${EXPERIMENT_NAME}}

########################### parameter groups ############################

DATA=(
    data.train_files="${TRAIN_FILE}"
    data.val_files="${VAL_FILES}"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.gen_batch_size=${GEN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.truncation=error
    data.image_key=images
    data.image_patch_size=16
    data.seed=42
    # Package import preserves class identity when HF filtering serializes it.
    data.custom_cls.path=pkg://recipe.papo.eval_dataset
    data.custom_cls.name=PAPOEvaluationDataset
    +data.papo_eval_max_pixels=1048576
)

MODEL=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.ppo_epochs=1
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW}
    actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH}
    actor_rollout_ref.actor.clip_ratio_c=10.0
    actor_rollout_ref.actor.loss_agg_mode=token-mean
    actor_rollout_ref.actor.fsdp_config.use_torch_compile=False
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.do_sample=True
    actor_rollout_ref.rollout.temperature=${TEMPERATURE}
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE}
    actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P}
    actor_rollout_ref.rollout.val_kwargs.top_k=${VAL_TOP_K}
    actor_rollout_ref.rollout.val_kwargs.presence_penalty=${VAL_PRESENCE_PENALTY}
    actor_rollout_ref.rollout.val_kwargs.n=1
    actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN}
    actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS}
    actor_rollout_ref.rollout.max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS}
    actor_rollout_ref.rollout.enforce_eager=True
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.disable_log_stats=False
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU}
)

DAPO=(
    algorithm.adv_estimator=grpo
    algorithm.norm_adv_by_std_in_grpo=True
    algorithm.use_kl_in_reward=False
    algorithm.filter_groups.enable=True
    algorithm.filter_groups.metric=acc
    algorithm.filter_groups.max_num_gen_batches=${MAX_NUM_GEN_BATCHES}
    distillation.enabled=False
    reward.reward_manager.name=dapo
    reward.custom_reward_function.path=recipe/papo/eval_reward.py
    reward.custom_reward_function.name=compute_score
    reward.reward_kwargs.max_resp_len=${MAX_RESPONSE_LENGTH}
    reward.reward_kwargs.overlong_buffer_cfg.enable=True
    reward.reward_kwargs.overlong_buffer_cfg.len=${OVERLONG_BUFFER_LEN}
    reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=${OVERLONG_PENALTY_FACTOR}
    reward.reward_kwargs.overlong_buffer_cfg.log=True
)

PAPO=(
    +papo=default
    actor_rollout_ref.actor.fsdp_config.papo.enabled=${PAPO_ENABLED}
    actor_rollout_ref.actor.fsdp_config.papo.gradient_mode=${PAPO_GRADIENT_MODE}
    actor_rollout_ref.actor.fsdp_config.papo.mask_probability=${PAPO_MASK_PROBABILITY}
    actor_rollout_ref.actor.fsdp_config.papo.perception_coef=${PAPO_PERCEPTION_COEF}
    actor_rollout_ref.actor.fsdp_config.papo.clean_entropy_coef=${PAPO_CLEAN_ENTROPY_COEF}
    actor_rollout_ref.actor.fsdp_config.papo.masked_entropy_coef=${PAPO_MASKED_ENTROPY_COEF}
)

TRAINER=(
    trainer.use_v1=False
    trainer.balance_batch=True
    trainer.nnodes=1
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.logger='["console","tensorboard"]'
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.val_before_train=True
    trainer.total_epochs=${TOTAL_EPOCHS}
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS}
    trainer.test_freq=${TEST_FREQ}
    trainer.save_freq=${SAVE_FREQ}
    trainer.resume_mode=disable
    trainer.default_local_dir="${ARTIFACT_DIR}/checkpoints"
    trainer.max_actor_ckpt_to_keep=8
    trainer.rollout_data_dir="${ARTIFACT_DIR}/rollouts"
    trainer.validation_data_dir="${ARTIFACT_DIR}/validation"
)

############################## launch ###################################
exec "${PAPO_PYTHON}" -m examples.on_policy_distillation_trainer.dapo_geo3k.main_papo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${DAPO[@]}" \
    "${TRAINER[@]}" \
    "${PAPO[@]}" \
    +ray_kwargs.ray_init.address=local \
    "$@"
