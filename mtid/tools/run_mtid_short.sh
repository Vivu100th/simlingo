#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MTID_EXPERIMENT="${MTID_EXPERIMENT:-mtid_short}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "${REPO_ROOT}/simlingo_training"
exec "${PYTHON_BIN}" train.py "experiment=${MTID_EXPERIMENT}" "$@"
