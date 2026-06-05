#!/usr/bin/env bash
set -euo pipefail

EPISODES="${1:-100}"
OUT_DIR="${2:-results_stage2/curriculum_probe}"
SPEED_SWEEP="${SPEED_SWEEP:-0.0,0.05,0.10,0.15}"

mkdir -p "${OUT_DIR}"

python -m stage2.diagnose \
  --config config_stage2_target_narrow_probe.yaml \
  --scenario randomized \
  --episodes "${EPISODES}" \
  --speed-sweep "${SPEED_SWEEP}" \
  | tee "${OUT_DIR}/target_narrow_guide_diagnose.json"

python -m stage2.diagnose \
  --config config_stage2_target_centered_probe.yaml \
  --scenario randomized \
  --episodes "${EPISODES}" \
  --speed-sweep "${SPEED_SWEEP}" \
  | tee "${OUT_DIR}/target_centered_guide_diagnose.json"

echo "curriculum probe outputs written to ${OUT_DIR}"
