#!/usr/bin/env bash
# Cached PAPO-D on two GPUs; preserve the global batch and optimizer schedule.
# Offload between phases so full-parameter training can share GPUs with vLLM.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export NGPUS_PER_NODE=2
export PAPO_GRADIENT_MODE=cached_old
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_qwen3_vl_4b_geo3k_papo_4gpu.sh" \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    "$@"
