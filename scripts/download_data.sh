#!/usr/bin/env bash
# Pre-fetch benchmark datasets into the HuggingFace cache.
#
# Usage:
#   bash scripts/download_data.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

PYTHON="${PYTHON:-$PROJECT_ROOT/.venv/bin/python3}"
if [ ! -x "$PYTHON" ]; then
    PYTHON="$(command -v python3)"
fi

echo "=== Pre-fetching benchmark datasets into HF cache ==="
echo "Python: $PYTHON"
echo

"$PYTHON" - <<'PYEOF'
import os
os.environ.setdefault("HF_DATASETS_OFFLINE", "0")
from datasets import load_dataset

for split in ("full_context", "medium_context", "small_context"):
    print(f"  • RepoExec (split: {split}) ...", flush=True)
    load_dataset("Fsoft-AIC/RepoExec", split=split)
    print("    done")

print(f"  • SWE-bench Verified (split: test) ...", flush=True)
load_dataset("SWE-bench/SWE-bench_Verified", split="test")
print("    done")

print("\nAll HuggingFace datasets cached.")
PYEOF

echo
echo "=== Done ==="
