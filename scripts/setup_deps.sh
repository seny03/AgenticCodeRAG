#!/usr/bin/env bash
set -euo pipefail

VENDOR_DIR="$(cd "$(dirname "$0")/.." && pwd)/vendor"
mkdir -p "$VENDOR_DIR"

echo "=== Cloning benchmark and tool repositories into $VENDOR_DIR ==="

clone_at_commit() {
    local url="$1"
    local dir="$2"
    local commit="$3"
    if [ -d "$dir/.git" ]; then
        echo "Pinning $dir to $commit"
        git -C "$dir" fetch --quiet
        git -C "$dir" checkout --quiet "$commit"
    else
        echo "Cloning $url -> $dir"
        git clone "$url" "$dir"
        git -C "$dir" checkout --quiet "$commit"
    fi
}

clone_at_commit \
    https://github.com/nju-websoft/DraCo.git \
    "$VENDOR_DIR/DraCo" \
    "7d558cd1da2388a7bf1b8b0a30bc4e6fd47cdf83"

clone_at_commit \
    https://github.com/FSoft-AI4Code/RepoExec.git \
    "$VENDOR_DIR/RepoExec" \
    "3998bb5170a2ee8174eb8e9ca1eebf6f85f07b55"

clone_at_commit \
    https://github.com/SWE-bench/SWE-bench.git \
    "$VENDOR_DIR/SWE-bench" \
    "f7bbbb2ccdf479001d6467c9e34af59e44a840f9"

echo ""
echo "=== Patching vendor files ==="

REPOEXEC_DOCKERFILE="$VENDOR_DIR/RepoExec/execution-code-eval/Dockerfile"
if [ -f "$REPOEXEC_DOCKERFILE" ] && grep -q "pip install -e /codegendata/human-eval" "$REPOEXEC_DOCKERFILE"; then
    echo "Patching $REPOEXEC_DOCKERFILE"
    sed -i 's|RUN pip install -e /codegendata/human-eval|RUN pip install tqdm fire numpy \\\n    \&\& pip install /codegendata/human-eval|' "$REPOEXEC_DOCKERFILE"
fi

HUMANEVAL_SETUP="$VENDOR_DIR/RepoExec/execution-code-eval/human-eval/setup.py"
if [ -f "$HUMANEVAL_SETUP" ] && grep -q "pkg_resources.parse_requirements" "$HUMANEVAL_SETUP"; then
    echo "Patching $HUMANEVAL_SETUP"
    python3 - "$HUMANEVAL_SETUP" <<'PYEOF'
import sys
path = sys.argv[1]
src = open(path).read()
src = src.replace("import pkg_resources\n", "")
src = src.replace(
    "        str(r)\n        for r in pkg_resources.parse_requirements(\n            open(os.path.join(os.path.dirname(__file__), \"requirements.txt\"))\n        )",
    "        r.strip()\n        for r in open(os.path.join(os.path.dirname(__file__), \"requirements.txt\"))\n        if r.strip() and not r.startswith(\"#\")",
)
open(path, "w").write(src)
print("  done")
PYEOF
fi

PROCESS_RESULT="$VENDOR_DIR/RepoExec/execution-code-eval/process_result.py"
if [ -f "$PROCESS_RESULT" ] && grep -q '"short_context"' "$PROCESS_RESULT"; then
    echo "Patching $PROCESS_RESULT (add small_context to choices)"
    sed -i 's/choices=\["full_context", "medium_context", "short_context"\]/choices=["full_context", "medium_context", "short_context", "small_context"]/' "$PROCESS_RESULT"
fi

echo ""
echo "=== Installing DraCo dependencies ==="
if [ -f "$VENDOR_DIR/DraCo/requirements.txt" ]; then
    grep -v "^transformers" "$VENDOR_DIR/DraCo/requirements.txt" | pip install -r /dev/stdin
    pip install "transformers>=4.40.0"
fi

echo ""
echo "=== Building RepoExec Docker image (codeeval-runner) ==="
REPOEXEC_EXEC_DIR="$VENDOR_DIR/RepoExec/execution-code-eval"
if [ -f "$REPOEXEC_EXEC_DIR/Dockerfile" ]; then
    if command -v docker &>/dev/null; then
        echo "Building codeeval-runner image (this may take a few minutes)..."
        (cd "$REPOEXEC_EXEC_DIR" && sudo docker build -t codeeval-runner -f Dockerfile --platform linux/amd64 . 2>&1)
        echo "Docker image built: codeeval-runner"
    else
        echo "Docker not found — skipping image build. Install Docker to enable RepoExec execution."
    fi
fi

echo ""
echo "=== Extracting RepoExec test-apps ==="
TESTAAPS_ZIP="$VENDOR_DIR/RepoExec/test-apps.zip"
TESTAPPS_DIR="$VENDOR_DIR/RepoExec/test-apps"
if [ -f "$TESTAAPS_ZIP" ] && [ ! -d "$TESTAPPS_DIR" ]; then
    echo "Extracting test-apps.zip..."
    (cd "$VENDOR_DIR/RepoExec" && unzip -q test-apps.zip)
    echo "Done."
fi

echo ""
echo "=== Installing SWE-bench ==="
pip install -e "$VENDOR_DIR/SWE-bench"

echo ""
echo "=== Installing this package ==="
pip install -e "$(cd "$(dirname "$0")/.." && pwd)"

echo ""
echo "=== Pre-downloading sentence-transformers model (all-MiniLM-L6-v2) ==="
HF_XET_ENABLED=0 python3 -c "
from sentence_transformers import SentenceTransformer
print('Downloading all-MiniLM-L6-v2...')
SentenceTransformer('all-MiniLM-L6-v2')
print('Done.')
"

echo ""
echo "=== Done ==="
echo "Vendor repos are in: $VENDOR_DIR"
