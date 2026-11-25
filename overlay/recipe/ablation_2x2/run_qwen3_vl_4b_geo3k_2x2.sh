#!/usr/bin/env bash
# Run from the pinned veRL checkout with this overlay applied.
set -euo pipefail

arm=${1:-}
case "$arm" in
    A) occlusion=0.0; evidence=false; routing=false ;;
    B) occlusion=${PAPO_PERCEPTION_COEF:-0.01}; evidence=false; routing=false ;;
    C) occlusion=0.0; evidence=true; routing=true ;;
    D) occlusion=${PAPO_PERCEPTION_COEF:-0.01}; evidence=true; routing=true ;;
    C-reward-only) occlusion=0.0; evidence=true; routing=false ;;
    *)
        printf 'Usage: %s {A|B|C|D|C-reward-only} [Hydra overrides or flags]\n' "${0##*/}" >&2
        exit 2
        ;;
esac
shift

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root"
seed=${TRAIN_SEED:-42}
label=$arm
if [[ "$arm" == B && "$occlusion" =~ ^0(\.0+)?$ ]]; then
    label=B-clean-nll
fi
name="${label}-seed${seed}-${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
data=${MOCA_DATA:-${root}/data/moca}
run_dir="${ABLATION_OUTPUT_DIR:-${root}/outputs/ablation_2x2}/${name}"

export RAY_ADDRESS=local
export TOKENIZERS_PARALLELISM=false
export NCCL_CUMEM_HOST_ENABLE=${NCCL_CUMEM_HOST_ENABLE:-0}

exec "${ABLATION_PYTHON:-python}" -m recipe.ablation_2x2.main_ablation \
    "data.train_files=$data/train.parquet" \
    "data.val_files=[$data/validation-1.parquet,$data/validation-2.parquet,$data/validation-3.parquet,$data/validation-4.parquet]" \
    "data.seed=$seed" \
    "actor_rollout_ref.model.path=${MODEL_PATH:-Qwen/Qwen3-VL-4B-Instruct}" \
    "actor_rollout_ref.actor.fsdp_config.seed=$seed" \
    "actor_rollout_ref.actor.fsdp_config.papo.perception_coef=$occlusion" \
    "moca.use_evidence_reward=$evidence" \
    "moca.verifier.base_url=${MOCA_VERIFIER_URL:-http://127.0.0.1:8000/v1}" \
    "moca.verifier.model=${MOCA_VERIFIER_MODEL:-verifier}" \
    "ablation.route_tokens=$routing" \
    "trainer.n_gpus_per_node=${NGPUS_PER_NODE:-2}" \
    "trainer.experiment_name=$name" \
    "trainer.default_local_dir=$run_dir/checkpoints" \
    "trainer.rollout_data_dir=$run_dir/rollouts" \
    "trainer.validation_data_dir=$run_dir/validation" \
    "$@"
