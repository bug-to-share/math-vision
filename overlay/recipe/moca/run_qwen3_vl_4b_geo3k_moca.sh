#!/usr/bin/env bash
# Independent full-parameter MoCA pilot. Called by the two-/four-GPU wrappers.
set -euo pipefail
MOCA_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "${MOCA_ROOT}"
MOCA_PYTHON=${MOCA_PYTHON:-python}
MOCA_DATA=${MOCA_DATA:-${MOCA_ROOT}/data/moca}
MOCA_RUN_TAG=${MOCA_RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}
MOCA_RUN_DIR=${MOCA_RUN_DIR:-${MOCA_ROOT}/outputs/moca/pilot-${MOCA_RUN_TAG}}
MOCA_DRY_RUN=${MOCA_DRY_RUN:-0}
export MOCA_NUM_GPUS=${MOCA_NUM_GPUS:-2}
export MOCA_SHARED_GPU_PIDS=${MOCA_SHARED_GPU_PIDS:-}
export MOCA_RUN_DIR

if [[ "${MOCA_DRY_RUN}" != 1 ]]; then
    : "${CUDA_VISIBLE_DEVICES:?Select released physical GPUs explicitly before launching}"
    "${MOCA_PYTHON}" - <<'PY'
import os, subprocess, json, pathlib
devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
expected = int(os.environ['MOCA_NUM_GPUS'])
if expected not in (2, 4) or len(devices) != expected or len(set(devices)) != expected or not all(x.isdigit() for x in devices):
    raise SystemExit(f'This pilot requires {expected} distinct physical GPU indices')
shared = {int(x) for x in os.environ['MOCA_SHARED_GPU_PIDS'].split(',') if x}
snapshot = []
for device in devices:
    used, total = map(int, subprocess.check_output([
        'nvidia-smi', '-i', device, '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'
    ], text=True).strip().split(','))
    pids = {int(x.strip()) for x in subprocess.check_output([
        'nvidia-smi', '-i', device, '--query-compute-apps=pid', '--format=csv,noheader,nounits'
    ], text=True).splitlines() if x.strip().isdigit()}
    if pids - shared:
        raise SystemExit(f'GPU {device} contains unapproved processes: {sorted(pids - shared)}')
    if used > 2048 and (not pids or not pids <= shared or used > total * 0.10):
        raise SystemExit(f'GPU {device} already uses {used} MiB; release it before launching')
    snapshot.append({'gpu': int(device), 'used_mib': used, 'total_mib': total, 'shared_pids': sorted(pids)})
path = pathlib.Path(os.environ['MOCA_RUN_DIR'])
path.mkdir(parents=True, exist_ok=True)
(path / 'gpu-preflight.json').write_text(json.dumps(snapshot, indent=2) + '\n')
PY
    test -f "${MOCA_DATA}/manifest.json"
    mkdir -p "${MOCA_RUN_DIR}"
fi

export RAY_ADDRESS=local
export NCCL_CUMEM_HOST_ENABLE=${NCCL_CUMEM_HOST_ENABLE:-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export TOKENIZERS_PARALLELISM=false
export PATH="$(dirname "$(command -v "${MOCA_PYTHON}")"):${PATH}"
MOCA_EXTRA=()
if [[ "${MOCA_DRY_RUN}" == 1 ]]; then
    MOCA_EXTRA=(--cfg job --resolve)
fi
exec "${MOCA_PYTHON}" -m recipe.moca.main_moca \
    "${MOCA_EXTRA[@]}" \
    data.train_files="${MOCA_DATA}/train.parquet" \
    data.val_files="[${MOCA_DATA}/validation-1.parquet,${MOCA_DATA}/validation-2.parquet,${MOCA_DATA}/validation-3.parquet,${MOCA_DATA}/validation-4.parquet]" \
    data.custom_cls.path=recipe/moca/dataset.py \
    data.custom_cls.name=MoCADataset \
    data.train_batch_size="${TRAIN_BATCH_SIZE:-4}" \
    data.max_prompt_length=2048 \
    data.max_response_length="${MAX_RESPONSE_LENGTH:-8192}" \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=4 \
    data.truncation=error \
    data.image_key=images \
    data.image_patch_size=16 \
    data.seed=42 \
    actor_rollout_ref.model.path="${MODEL_PATH:-Qwen/Qwen3-VL-4B-Instruct}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_BATCH_SIZE:-4}" \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.use_torch_compile=False \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.n="${ROLLOUT_N:-8}" \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM_UTIL:-0.75}" \
    actor_rollout_ref.rollout.max_model_len="$((2048 + ${MAX_RESPONSE_LENGTH:-8192}))" \
    actor_rollout_ref.rollout.max_num_seqs=16 \
    actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.8 \
    actor_rollout_ref.rollout.val_kwargs.top_k=20 \
    actor_rollout_ref.rollout.val_kwargs.presence_penalty=1.5 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    moca.verifier.base_url="${MOCA_VERIFIER_URL:-http://127.0.0.1:8000/v1}" \
    moca.verifier.model="${MOCA_VERIFIER_MODEL:-verifier}" \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node="${MOCA_NUM_GPUS}" \
    trainer.total_epochs=2 \
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS:-20}" \
    trainer.val_before_train=False \
    trainer.test_freq="${TEST_FREQ:-20}" \
    trainer.save_freq="${SAVE_FREQ:-10}" \
    trainer.resume_mode=disable \
    trainer.logger='[console,tensorboard]' \
    trainer.experiment_name="moca-grpo-${MOCA_RUN_TAG}" \
    trainer.default_local_dir="${MOCA_RUN_DIR}/checkpoints" \
    trainer.rollout_data_dir="${MOCA_RUN_DIR}/rollouts" \
    trainer.validation_data_dir="${MOCA_RUN_DIR}/validation" \
    "$@"
