#!/usr/bin/env bash
# Matched to the recompute_grad experiment; only masked log-prob semantics change.
set -euo pipefail
export PAPO_GRADIENT_MODE=cached_old
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_qwen3_vl_4b_geo3k_papo_4gpu.sh" "$@"
