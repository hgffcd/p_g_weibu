#!/usr/bin/env bash
set -euo pipefail

OUT="${1:-mra_rlec_server_artifacts.tar.gz}"
items=()
for item in config*.yaml logs logs_vector checkpoints checkpoints_vector results server_runs server_runs_vector server_runs_vector_multi reproduction_results.md implementation_notes.md SERVER_TRAINING.md server_artifact_analysis.md assumptions.md; do
  if compgen -G "${item}" > /dev/null; then
    items+=(${item})
  fi
done

if [ "${#items[@]}" -eq 0 ]; then
  echo "no artifacts found"
  exit 1
fi

tar -czf "${OUT}" "${items[@]}"
echo "wrote ${OUT}"
