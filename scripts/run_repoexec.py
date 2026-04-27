#!/usr/bin/env python3
"""Run RepoExec evaluation."""

import os
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import ast
import contextlib
import argparse
import logging
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
load_dotenv()

from _utils import COMPLETION_SYSTEM_PROMPT, agent_cfg_from_setup, build_generate_fn, load_setup_config
from agentic_code_rag.agent.graph import build_agent_graph
from agentic_code_rag.agent.llm_provider import create_llm
from agentic_code_rag.agent.state import AgentState
from agentic_code_rag.benchmarks.repoexec import CONTEXT_LEVELS, _strip_markdown, run_repoexec
from agentic_code_rag.draco_wrapper import DraCoIndex, DraCoToolAPI

logger = logging.getLogger(__name__)


def _find_target_file(repo_root: Path, module: str) -> Path | None:
    """Convert dotted module name to file path, return None if not found."""
    if not module:
        return None
    candidate = repo_root / (module.replace(".", "/") + ".py")
    return candidate if candidate.exists() else None


def _strip_function_body(content: str, entry_point: str) -> str:
    """
    Replace the entry_point function with a `pass` stub:

        def name(args, ...) -> ReturnType:
            pass

    Decorators, docstring, and body are dropped.  The signature (with type
    annotations) is preserved so the agent can grep for `def name` and
    immediately know where the function lives — this avoids the agent burning
    turns searching for a non-existent definition.

    The docstring is dropped specifically so BM25 / DraCo retrieval cannot
    bias the model toward an "existing implementation" pattern (the docstring
    is essentially the task prompt and would dominate BM25 scores).
    """
    import copy

    try:
        tree = ast.parse(content)
    except SyntaxError:
        return content

    lines = content.splitlines(keepends=True)

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name != entry_point:
            continue

        first_lineno = (
            node.decorator_list[0].lineno
            if node.decorator_list
            else node.lineno
        )
        end_lineno = node.end_lineno

        # Build a stub: keep signature, drop everything else.
        stub_node = copy.deepcopy(node)
        stub_node.body = [ast.Pass()]
        stub_node.decorator_list = []
        stub_text = ast.unparse(stub_node)

        # Re-indent to the original column offset (e.g. for class methods).
        indent = " " * node.col_offset
        stub_indented = "\n".join(
            (indent + line if line else line)
            for line in stub_text.splitlines()
        ) + "\n"

        return (
            "".join(lines[: first_lineno - 1])
            + stub_indented
            + "".join(lines[end_lineno:])
        )

    return content  # entry_point not found in file


@contextlib.contextmanager
def _stripped_file(target_file: Path | None, entry_point: str):
    """Context manager: strip target function body on enter, restore on exit."""
    if target_file is None:
        yield
        return
    original = target_file.read_text()
    stripped = _strip_function_body(original, entry_point)
    target_file.write_text(stripped)
    try:
        yield
    finally:
        target_file.write_text(original)


_TEST_DIR_NAMES = {"test", "tests", "testing"}
_BACKUP_SUFFIXES = (".orig", ".bak", ".backup", ".old", ".save")


def _is_test_file(p: Path) -> bool:
    return p.suffix == ".py" and (
        p.name.startswith("test_") or p.name.endswith("_test.py")
    )


def _is_backup_file(p: Path) -> bool:
    name = p.name
    if name.endswith("~"):
        return True
    if any(name.endswith(s) for s in _BACKUP_SUFFIXES):
        return True
    # Compound suffixes like manipulation.py.orig / file.py.bak
    parts = name.split(".")
    if len(parts) >= 3 and "." + parts[-1] in _BACKUP_SUFFIXES:
        return True
    return False


def _find_artifacts_to_hide(repo_root: Path) -> list[Path]:
    """
    Return paths to hide from the agent during a run:
      - test directories and loose test files (so the agent can't peek at
        evaluation tests)
      - backup files (.orig/.bak/.backup/~) anywhere in the repo (so the
        agent can't read leaked gold solutions)
    """
    results: list[Path] = []
    covered: set[Path] = set()

    for p in sorted(repo_root.rglob("*")):
        # Skip anything already inside a directory we're hiding
        if any(p == tp or str(p).startswith(str(tp) + "/") for tp in covered):
            continue
        if p.is_dir() and p.name.lower() in _TEST_DIR_NAMES:
            results.append(p)
            covered.add(p)
        elif p.is_file() and (_is_test_file(p) or _is_backup_file(p)):
            results.append(p)

    return results


@contextlib.contextmanager
def _hidden_artifacts(repo_root: Path):
    """
    Move test files/dirs and backup files outside the repo for the agent run,
    restore them afterwards.

    Hides:
      - test/ tests/ testing/ directories
      - test_*.py / *_test.py files
      - *.orig *.bak *.backup *.old *.save *~ files (potential gold leaks)
    """
    paths = _find_artifacts_to_hide(repo_root)
    if not paths:
        yield
        return

    tmp_dir = Path(tempfile.mkdtemp(prefix="repoexec_hidden_"))
    moved: list[tuple[Path, Path]] = []
    try:
        for orig in paths:
            rel = orig.relative_to(repo_root)
            backup = tmp_dir / rel
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(orig), str(backup))
            moved.append((orig, backup))
        yield
    finally:
        for orig, backup in moved:
            orig.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(backup), str(orig))
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base_url", required=True)
    parser.add_argument("--output", default="results/repoexec")
    parser.add_argument("--context_level", default="medium_context", choices=CONTEXT_LEVELS)
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--k_values", nargs="+", type=int, default=[1])
    parser.add_argument("--max_samples", type=int, default=None)
    _default_repoexec = str(Path(__file__).resolve().parents[1] / "vendor" / "RepoExec")
    parser.add_argument("--repoexec_repo", default=_default_repoexec)
    parser.add_argument("--setup", default=None, help="Path to a setup YAML (configs/setup_*.yaml)")
    parser.add_argument("--max_tokens", type=int, default=None,
                        help="Max output tokens (default: llm_provider default of 16384)")
    parser.add_argument("--max_workers", type=int, default=1,
                        help="Number of parallel workers for task generation (default: 1, sequential)")
    parser.add_argument("--log_level", default=os.environ.get("LOG_LEVEL", "INFO"),
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / "run.log"
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stderr),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )

    setup = load_setup_config(args.setup)
    mode = setup.get("mode", "agent")

    llm_kwargs = {}
    if args.max_tokens is not None:
        llm_kwargs["max_tokens"] = args.max_tokens
    llm = create_llm(args.provider, args.model, base_url=args.base_url, **llm_kwargs)
    trajectory_dir = output_dir / "trajectories"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    agent_cfg = agent_cfg_from_setup(setup)

    def _make_repo_copy(src: Path, task_id: str) -> Path:
        """Copy `src` to a unique tmp dir for this task.  Caller must rmtree(parent)."""
        tmp_parent = Path(tempfile.mkdtemp(prefix=f"repoexec_task{task_id}_"))
        dst = tmp_parent / src.name
        shutil.copytree(src, dst, symlinks=True)
        return dst

    def _build_tool_api(repo_root: Path, target_file: Path | None,
                        entry_point: str) -> DraCoToolAPI:
        """Build a DraCo index for `repo_root` while the target function is stripped.

        The graph is written to a per-copy directory so concurrent tasks don't
        clobber each other's caches.  No cross-task on-disk caching — each
        parallel task uses its own throwaway copy.
        """
        graph_dir = repo_root.parent / "_draco_graph"
        with _stripped_file(target_file, entry_point):
            index = DraCoIndex(repo_root, graph_dir=graph_dir)
            try:
                index.build()
            except Exception as exc:
                logger.warning("DraCo index failed: %s", exc)
        return DraCoToolAPI(index, repo_root)

    def _save_simple_trajectory(
        task_traj_dir: Path,
        prompt: str,
        traj_mode: str,
        augmented_prompt: str = "",
        raw_output: str = "",
        postprocessed_output: str = "",
        final_output: str = "",
    ) -> None:
        task_traj_dir.mkdir(parents=True, exist_ok=True)
        stem = task_traj_dir.name
        sections = [
            f"# Trajectory: {stem}",
            f"",
            f"**Mode:** {traj_mode}",
        ]

        # Inputs
        sections += [
            f"",
            f"---",
            f"## 1. System Prompt",
            f"```",
            COMPLETION_SYSTEM_PROMPT.strip(),
            f"```",
            f"",
            f"## 2. User Prompt (original task)",
            f"```",
            prompt.strip(),
            f"```",
        ]
        if augmented_prompt and augmented_prompt != prompt:
            sections += [
                f"",
                f"## 3. Augmented Prompt (sent to LLM)",
                f"```",
                augmented_prompt.strip(),
                f"```",
            ]

        # Outputs
        step = 4 if (augmented_prompt and augmented_prompt != prompt) else 3

        sections += [
            f"",
            f"---",
            f"## {step}. Raw Model Output",
            f"```",
            raw_output.strip(),
            f"```",
        ]
        step += 1

        if postprocessed_output.strip() != raw_output.strip():
            sections += [
                f"",
                f"## {step}. After postprocess_completion (strip fences + prepend def if missing)",
                f"```python",
                postprocessed_output.strip(),
                f"```",
            ]
            step += 1

        if final_output.strip() != postprocessed_output.strip():
            sections += [
                f"",
                f"## {step}. After _strip_markdown (outer fence removal)",
                f"```python",
                final_output.strip(),
                f"```",
            ]
            step += 1

        if final_output.strip() == postprocessed_output.strip() == raw_output.strip():
            # Nothing changed
            sections[-4] = f"## {step - 1}. Raw Model Output (no post-processing needed)"

        (task_traj_dir / f"{stem}.md").write_text("\n".join(sections) + "\n")

    def make_generate_fn(
        repo_root: Path,
        task_id: str,
        entry_point: str,
        module: str,
        target_function_prompt: str,
    ) -> Callable[[str], str]:
        repo_root = repo_root.resolve()
        safe_entry = entry_point.replace("/", "_") or "task"
        task_traj_dir = trajectory_dir / f"{task_id}_{repo_root.name}_{safe_entry}"
        target_file = _find_target_file(repo_root, module)

        if mode == "zero_shot":
            # No file access — no stripping needed on input.
            raw_cap: list = []
            fn = build_generate_fn(llm, setup, tool_api=None, raw_capture=raw_cap)
            assert fn is not None
            def zero_shot_fn(prompt: str) -> str:
                output = fn(prompt)
                raw_out = raw_cap[0] if raw_cap else output
                final = _strip_markdown(output)
                _save_simple_trajectory(
                    task_traj_dir, prompt, "zero_shot",
                    raw_output=raw_out,
                    postprocessed_output=output,
                    final_output=final,
                )
                return output
            return zero_shot_fn

        # For static_rag and agent: index build + inference happen inside a single
        # _stripped_file window so the gold implementation is never on disk during
        # any index construction or agent tool call.
        if mode == "static_rag":
            def static_rag_fn(prompt: str) -> str:
                context_capture: list = []
                raw_capture: list = []
                work_root = _make_repo_copy(repo_root, task_id)
                work_target = _find_target_file(work_root, module)
                try:
                    with _hidden_artifacts(work_root), _stripped_file(work_target, entry_point):
                        tool_api = _build_tool_api(work_root, work_target, entry_point)
                        fn = build_generate_fn(llm, setup, tool_api=tool_api,
                                              context_capture=context_capture,
                                              raw_capture=raw_capture)
                        assert fn is not None
                        output = fn(prompt)
                finally:
                    shutil.rmtree(work_root.parent, ignore_errors=True)
                augmented = context_capture[0] if context_capture else prompt
                raw_out = raw_capture[0] if raw_capture else output
                final = _strip_markdown(output)
                _save_simple_trajectory(
                    task_traj_dir, prompt, "static_rag",
                    augmented_prompt=augmented,
                    raw_output=raw_out,
                    postprocessed_output=output,
                    final_output=final,
                )
                return output
            return static_rag_fn

        # Agent mode
        task_traj_dir.mkdir(parents=True, exist_ok=True)

        def agent_fn(prompt: str) -> str:
            work_root = _make_repo_copy(repo_root, task_id)
            work_target = _find_target_file(work_root, module)
            try:
                with _hidden_artifacts(work_root), _stripped_file(work_target, entry_point):
                    tool_api = _build_tool_api(work_root, work_target, entry_point)
                    agent_graph = build_agent_graph(
                        tool_api=tool_api,
                        llm=llm,
                        config=agent_cfg,
                        trajectory_dir=task_traj_dir,
                    )
                    init_state = AgentState(
                        task_text=prompt,
                        repo_root=str(work_root),
                        snapshot_id=entry_point,
                        output_format="code",
                    )
                    final_state = agent_graph.invoke(init_state)
            finally:
                shutil.rmtree(work_root.parent, ignore_errors=True)
            answer = (
                final_state.get("proposed_patch")
                or final_state.get("retrieved_context")
                or ""
            )
            return answer if isinstance(answer, str) else str(answer)

        return agent_fn

    results = run_repoexec(
        make_generate_fn=make_generate_fn,
        output_dir=output_dir,
        repoexec_repo=Path(args.repoexec_repo),
        context_level=args.context_level,
        n_samples=args.n_samples,
        k_values=args.k_values,
        max_samples=args.max_samples,
        trajectory_dir=trajectory_dir,
        max_workers=args.max_workers,
    )
    if setup:
        print(f"Setup: {setup.get('name', args.setup)}")
    print("Results:", results)


if __name__ == "__main__":
    main()
