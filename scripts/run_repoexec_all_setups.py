#!/usr/bin/env python3
"""
Run RepoExec across all *.yaml configs in a folder.

Mirrors the interface of run_repoexec.py, replacing --setup with --setup_folder.
Each YAML is run as a separate subprocess; results go to output/<yaml_stem>/.

Example:
    python scripts/run_repoexec_all_setups.py \
        --provider anthropic \
        --model "claude-4-6" \
        --base_url "https://..." \
        --context_level small_context \
        --output results/repoexec_medium_context \
        --setup_folder configs/
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agentic_code_rag.benchmarks.repoexec import CONTEXT_LEVELS

RUNNER = Path(__file__).resolve().parent / "run_repoexec.py"


def main():
    parser = argparse.ArgumentParser(
        description="Run RepoExec for every *.yaml in --setup_folder."
    )
    parser.add_argument("--provider", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base_url", required=True)
    parser.add_argument("--output", default="results/repoexec",
                        help="Base output directory; each setup writes to output/<yaml_stem>/")
    parser.add_argument("--context_level", default="medium_context", choices=CONTEXT_LEVELS)
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--k_values", nargs="+", type=int, default=[1])
    parser.add_argument("--max_samples", type=int, default=None)
    _default_repoexec = str(Path(__file__).resolve().parents[1] / "vendor" / "RepoExec")
    parser.add_argument("--repoexec_repo", default=_default_repoexec)
    parser.add_argument("--setup_folder", required=True,
                        help="Folder containing *.yaml setup configs")
    parser.add_argument("--max_workers", type=int, default=1,
                        help="Parallel workers per setup (default: 1, sequential)")
    args = parser.parse_args()

    yamls = sorted(Path(args.setup_folder).glob("*.yaml"))
    if not yamls:
        print(f"No *.yaml files found in {args.setup_folder}", file=sys.stderr)
        sys.exit(1)

    output_base = Path(args.output)
    passed, failed = [], []

    for yaml_path in yamls:
        stem = yaml_path.stem
        output_dir = output_base / stem

        print(f"\n{'━' * 60}")
        print(f"  {stem}  ->  {output_dir}")
        print(f"{'━' * 60}", flush=True)

        cmd = [
            sys.executable, str(RUNNER),
            "--provider",      args.provider,
            "--model",         args.model,
            "--base_url",      args.base_url,
            "--context_level", args.context_level,
            "--output",        str(output_dir),
            "--setup",         str(yaml_path),
            "--n_samples",     str(args.n_samples),
            "--k_values",      *[str(k) for k in args.k_values],
            "--repoexec_repo", args.repoexec_repo,
        ]
        if args.max_samples is not None:
            cmd += ["--max_samples", str(args.max_samples)]
        if args.max_workers != 1:
            cmd += ["--max_workers", str(args.max_workers)]

        result = subprocess.run(cmd)
        if result.returncode == 0:
            passed.append(stem)
        else:
            failed.append(stem)
            print(f"  ERROR: {stem} exited with code {result.returncode}", file=sys.stderr)

    print(f"\n{'═' * 60}")
    print(f"Done: {len(passed)}/{len(yamls)} setups completed")
    if failed:
        print("Failed:")
        for name in failed:
            print(f"  - {name}")
        sys.exit(1)


if __name__ == "__main__":
    main()
