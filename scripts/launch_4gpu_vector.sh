#!/usr/bin/env bash
set -euo pipefail

BASE_CONFIG="${1:-config_vector_server.yaml}"
RESUME="${2:-}"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-60}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-600}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
mkdir -p server_runs_vector_multi
PIDS=()
LOGS=()
START_TIME="$(date +%s)"

for GPU in 0 1 2 3; do
  CFG="config_vector_gpu${GPU}.yaml"
  LOG="server_runs_vector_multi/train_gpu${GPU}.log"
  cp "${BASE_CONFIG}" "${CFG}"
  python - <<PY
import json
from pathlib import Path
cfg_path = Path("${CFG}")
cfg = json.loads(cfg_path.read_text())
cfg["training"]["seed"] = 7 + ${GPU} * 100
cfg["training"]["checkpoint_dir"] = f"checkpoints_vector/gpu${GPU}"
cfg["training"]["log_dir"] = f"logs_vector/gpu${GPU}"
cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
PY
  if [ -n "${RESUME}" ]; then
    CUDA_VISIBLE_DEVICES=${GPU} python main.py train-vector --config "${CFG}" --resume "${RESUME}" \
      > "${LOG}" 2>&1 &
  else
    CUDA_VISIBLE_DEVICES=${GPU} python main.py train-vector --config "${CFG}" \
      > "${LOG}" 2>&1 &
  fi
  PID="$!"
  PIDS+=("${PID}")
  LOGS+=("${LOG}")
  echo "gpu${GPU}: pid=${PID} log=${LOG}"
done

echo "launched 4 vectorized runs"
echo "monitoring every ${MONITOR_INTERVAL}s; press Ctrl+C only if you want to stop monitoring."
echo "startup timeout is ${STARTUP_TIMEOUT}s; if no update=1 appears by then, jobs will be stopped."

while true; do
  RUNNING=0
  STARTED=0
  for PID in "${PIDS[@]}"; do
    if kill -0 "${PID}" 2>/dev/null; then
      RUNNING=1
      break
    fi
  done

  echo "----- $(date '+%Y-%m-%d %H:%M:%S') latest training status -----"
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
    echo "gpu${IDX}: training finished successfully"
  else
    echo "gpu${IDX}: training failed with exit code ${CODE}; check ${LOGS[$IDX]}"
    STATUS=1
  fi
done
set -e

if [ "${STATUS}" -eq 0 ]; then
  echo "all vectorized runs finished"
else
  echo "one or more vectorized runs failed"
fi
exit "${STATUS}"
