#!/usr/bin/env bash
set -euo pipefail

BASE_CONFIG="${1:-config_stage2_channel_hard_v1_finetune.yaml}"
WEIGHTS="${2:?weights checkpoint is required}"
RUN_NAME="${3:-stage2_channel_hard_v1_finetune}"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-60}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-600}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

mkdir -p "server_runs_${RUN_NAME}" "stage2_generated_configs"
PIDS=()
LOGS=()
START_TIME="$(date +%s)"

for GPU in 0 1 2 3; do
  CFG="stage2_generated_configs/${RUN_NAME}_gpu${GPU}.yaml"
  LOG="server_runs_${RUN_NAME}/train_gpu${GPU}.log"
  cp "${BASE_CONFIG}" "${CFG}"
  python - <<PY
from pathlib import Path
import json
try:
    import yaml
except ModuleNotFoundError:
    yaml = None

cfg_path = Path("${CFG}")
text = cfg_path.read_text(encoding="utf-8")
cfg = yaml.safe_load(text) if yaml is not None else json.loads(text)
cfg["training"]["seed"] = int(cfg["training"].get("seed", 0)) + ${GPU} * 100
cfg["training"]["checkpoint_dir"] = f"checkpoints_${RUN_NAME}/gpu${GPU}"
cfg["training"]["log_dir"] = f"logs_${RUN_NAME}/gpu${GPU}"
if yaml is not None:
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
else:
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
PY
  CUDA_VISIBLE_DEVICES=${GPU} python -m stage2.train_channel_vector_weights --config "${CFG}" --weights "${WEIGHTS}" \
    > "${LOG}" 2>&1 &
  PID="$!"
  PIDS+=("${PID}")
  LOGS+=("${LOG}")
  echo "gpu${GPU}: pid=${PID} log=${LOG}"
done

echo "launched 4 stage2 channel weights-only fine-tune runs: ${RUN_NAME}"
echo "monitoring every ${MONITOR_INTERVAL}s; press Ctrl+C only if you want to stop monitoring."

while true; do
  RUNNING=0
  STARTED=0
  for PID in "${PIDS[@]}"; do
    if kill -0 "${PID}" 2>/dev/null; then
      RUNNING=1
      break
    fi
  done

  echo "----- $(date '+%Y-%m-%d %H:%M:%S') latest fine-tune status -----"
  for IDX in "${!LOGS[@]}"; do
    LOG="${LOGS[$IDX]}"
    if [ -s "${LOG}" ]; then
      printf "gpu%s: " "${IDX}"
      STATUS_LINE="$(grep -E "update=|checkpoint=|best_checkpoint=|Traceback|Error|FileNotFoundError|ValueError|RuntimeError" "${LOG}" | tail -n 1 || true)"
      if [ -n "${STATUS_LINE}" ]; then
        echo "${STATUS_LINE}"
      else
        echo "log exists, waiting for first training update"
      fi
      if grep -q "update=1" "${LOG}"; then
        STARTED=1
      fi
    else
      echo "gpu${IDX}: log not written yet"
    fi
  done

  NOW="$(date +%s)"
  ELAPSED=$((NOW - START_TIME))
  if [ "${STARTED}" -eq 0 ] && [ "${ELAPSED}" -gt "${STARTUP_TIMEOUT}" ]; then
    echo "no update=1 after ${ELAPSED}s; stopping runs to avoid a stuck server job"
    for PID in "${PIDS[@]}"; do
      kill "${PID}" 2>/dev/null || true
    done
    sleep 5
    for PID in "${PIDS[@]}"; do
      kill -9 "${PID}" 2>/dev/null || true
    done
    exit 1
  fi

  if [ "${RUNNING}" -eq 0 ]; then
    break
  fi
  sleep "${MONITOR_INTERVAL}"
done

STATUS=0
set +e
for IDX in "${!PIDS[@]}"; do
  wait "${PIDS[$IDX]}"
  CODE="$?"
  if [ "${CODE}" -eq 0 ]; then
    echo "gpu${IDX}: fine-tune finished successfully"
  else
    echo "gpu${IDX}: fine-tune failed with exit code ${CODE}; check ${LOGS[$IDX]}"
    STATUS=1
  fi
done
set -e
exit "${STATUS}"
