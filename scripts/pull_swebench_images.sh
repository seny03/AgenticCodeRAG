#!/usr/bin/env bash
# Pull SWE-bench Verified Docker evaluation environments using the official harness.
#
# Required only when running run_swebench.py with --run_harness --dataset verified.
#
# Usage:
#   bash scripts/pull_swebench_images.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

PYTHON="${PYTHON:-$PROJECT_ROOT/.venv/bin/python3}"
if [ ! -x "$PYTHON" ]; then
    PYTHON="$(command -v python3)"
fi

MAX_WORKERS="${MAX_WORKERS:-4}"

echo "=== Pulling SWE-bench Verified Docker images (workers=${MAX_WORKERS}) ==="
"$PYTHON" -m swebench.harness.prepare_images \
    --dataset_name SWE-bench/SWE-bench_Verified \
    --split test \
    --max_workers "$MAX_WORKERS"
