#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-config_server.yaml}"
LOG_DIR="${2:-server_runs}"
RESUME="${3:-}"
mkdir -p "${LOG_DIR}"

python - <<'PY' > "${LOG_DIR}/server_probe.txt"
import sys, torch, numpy, yaml
print("python", sys.version)
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_version", torch.version.cuda)
print("device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device_name", torch.cuda.get_device_name(0))
print("numpy", numpy.__version__)
print("pyyaml", yaml.__version__)
PY

nvidia-smi > "${LOG_DIR}/nvidia-smi.txt" || true
conda list > "${LOG_DIR}/conda-list.txt" || true

if [ -n "${RESUME}" ]; then
  python main.py train --config "${CONFIG}" --resume "${RESUME}" 2>&1 | tee "${LOG_DIR}/train_stdout.log"
else
  python main.py train --config "${CONFIG}" 2>&1 | tee "${LOG_DIR}/train_stdout.log"
fi
