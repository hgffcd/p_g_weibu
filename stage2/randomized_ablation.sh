#!/usr/bin/env bash
set -euo pipefail

EPISODES="${1:-100}"
OUT_DIR="${2:-results_stage2/randomized_ablation}"
SPEED_SWEEP="${SPEED_SWEEP:-0.0,0.05,0.10,0.15}"

mkdir -p "${OUT_DIR}"

python -m stage2.diagnose \
  --config config_stage2_target_only_probe.yaml \
  --scenario randomized \
  --episodes "${EPISODES}" \
  --speed-sweep "${SPEED_SWEEP}" \
  | tee "${OUT_DIR}/target_only_guide_diagnose.json"

python -m stage2.diagnose \
  --config config_stage2_randomized_easy.yaml \
  --scenario randomized \
  --episodes "${EPISODES}" \
  --speed-sweep "${SPEED_SWEEP}" \
  | tee "${OUT_DIR}/easy_guide_diagnose.json"

if [[ -f checkpoints_stage2_moving_form/gpu1/mra_rlec_best.pt ]]; then
  python -m stage2.evaluate \
    --config config_stage2_randomized_easy.yaml \
    --scenario randomized \
    --policy checkpoint \
    --checkpoint checkpoints_stage2_moving_form/gpu1/mra_rlec_best.pt \
    --episodes "${EPISODES}" \
    --output-dir "${OUT_DIR}/moving_checkpoint_on_easy_randomized"
fi

echo "randomized ablation outputs written to ${OUT_DIR}"
