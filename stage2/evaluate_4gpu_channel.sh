#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${1:-stage2_channel}"
SCENARIO="${2:-randomized}"
EPISODES="${3:-300}"
OUT_ROOT="${4:-results_stage2/${RUN_NAME}_eval}"

mkdir -p "${OUT_ROOT}"

for GPU in 0 1 2 3; do
  CFG="stage2_generated_configs/${RUN_NAME}_gpu${GPU}.yaml"
  CKPT="checkpoints_${RUN_NAME}/gpu${GPU}/mra_rlec_best.pt"
  OUT_DIR="${OUT_ROOT}/gpu${GPU}"
  if [[ ! -f "${CFG}" ]]; then
    echo "missing config: ${CFG}" >&2
    exit 1
  fi
  if [[ ! -f "${CKPT}" ]]; then
    echo "missing checkpoint: ${CKPT}" >&2
    exit 1
  fi
  python -m stage2.channel_evaluate \
    --config "${CFG}" \
    --scenario "${SCENARIO}" \
    --policy checkpoint \
    --checkpoint "${CKPT}" \
    --episodes "${EPISODES}" \
    --output-dir "${OUT_DIR}"
done
