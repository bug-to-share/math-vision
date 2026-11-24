#!/usr/bin/env bash
set -euo pipefail
export MOCA_NUM_GPUS=2
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_qwen3_vl_4b_geo3k_moca.sh" "$@"
