#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${1:-stage2_moving}"
SCENARIO="${2:-fixed}"
EPISODES="${3:-200}"
OUT_ROOT="${4:-results_stage2/${RUN_NAME}_${SCENARIO}_eval}"

mkdir -p "${OUT_ROOT}"
for GPU in 0 1 2 3; do
  CFG="stage2_generated_configs/${RUN_NAME}_gpu${GPU}.yaml"
  if [ ! -f "${CFG}" ]; then
    CFG="config_stage2_moving.yaml"
  fi
  BEST="checkpoints_${RUN_NAME}/gpu${GPU}/mra_rlec_best.pt"
  LATEST="checkpoints_${RUN_NAME}/gpu${GPU}/mra_rlec_latest.pt"
  if [ -f "${BEST}" ]; then
    CKPT="${BEST}"
  else
    CKPT="${LATEST}"
  fi
  CUDA_VISIBLE_DEVICES=${GPU} python -m stage2.evaluate \
    --config "${CFG}" \
    --scenario "${SCENARIO}" \
    --policy checkpoint \
    --checkpoint "${CKPT}" \
    --episodes "${EPISODES}" \
    --output-dir "${OUT_ROOT}/gpu${GPU}"
done
