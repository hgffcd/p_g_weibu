#!/usr/bin/env bash
set -euo pipefail

EPISODES="${1:-100}"
OUT_ROOT="${2:-results/vector_eval}"
mkdir -p "${OUT_ROOT}"

for GPU in 0 1 2 3; do
  CFG="config_vector_gpu${GPU}.yaml"
  BEST="checkpoints_vector/gpu${GPU}/mra_rlec_best.pt"
  LATEST="checkpoints_vector/gpu${GPU}/mra_rlec_latest.pt"
  if [ -f "${BEST}" ]; then
    CKPT="${BEST}"
  else
    CKPT="${LATEST}"
  fi
  CUDA_VISIBLE_DEVICES=${GPU} python main.py metrics \
    --config "${CFG}" \
    --policy checkpoint \
    --checkpoint "${CKPT}" \
    --episodes "${EPISODES}" \
    --output-dir "${OUT_ROOT}/gpu${GPU}"
done
