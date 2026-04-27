"""
RepoExec runner with full Docker-based execution pipeline.

Pipeline: generate → generations.json → process_result.py →
          execute.py (Docker) → passk.py → pass@k
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Optional

from tqdm import tqdm

from ._hf_utils import load_dataset_cached

logger = logging.getLogger(__name__)

CONTEXT_LEVELS = ["full_context", "medium_context", "small_context"]


def run_repoexec(
    make_generate_fn: Callable[[Path, str, str, str, str], Callable[[str], str]],
    output_dir: Path,
    repoexec_repo: Path,
    context_level: str = "medium_context",
    n_samples: int = 1,
    k_values: Optional[list[int]] = None,
    max_samples: Optional[int] = None,
    trajectory_dir: Optional[Path] = None,
    max_workers: int = 1,
) -> dict:
    """
    Run RepoExec evaluation.

    Parameters
    ----------
    make_generate_fn:
        Callable(repo_root, task_id, entry_point, module, target_function_prompt)
        -> Callable(prompt) -> str.
        Called once per task. task_id and entry_point come from the dataset and
        are used for trajectory directory naming. The module and
        target_function_prompt fields are used to strip the gold implementation
        from the repo before building the retrieval index and running the agent.
    output_dir:
        Where to write predictions and results.
    repoexec_repo:
        Path to the cloned RepoExec vendor repo (execution-code-eval and
        extracted test-apps/ live here).
    context_level:
        One of full_context, medium_context, small_context.
    n_samples:
        Number of completions per problem (for pass@k).
    k_values:
        List of k values for pass@k. Defaults to [1].
    max_samples:
        Limit number of samples.
    """
    if context_level not in CONTEXT_LEVELS:
        raise ValueError(f"context_level must be one of {CONTEXT_LEVELS}")

    k_values = k_values or [1]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading RepoExec dataset: %s", context_level)
    dataset = load_dataset_cached("Fsoft-AIC/RepoExec", split=context_level)

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    # 1. Generate predictions (per-task).  LLM-level caching (if enabled) is
    # handled inside the LLM wrapper, transparent to this loop.
    def _process_one(loop_idx: int, sample: dict) -> tuple[list[str], Optional[Path]]:
        project = sample["project"]
        repo_root = (repoexec_repo / project).resolve()
        traj_path: Optional[Path] = None
        if trajectory_dir is not None:
            entry_point_raw = sample.get("entry_point", "")
            safe_entry = entry_point_raw.replace("/", "_") or "task"
            task_id_str = str(sample.get("id", ""))
            traj_path = trajectory_dir / f"{task_id_str}_{repo_root.name}_{safe_entry}"

        if not repo_root.exists():
            logger.warning("Project missing on disk, using empty prediction: %s", repo_root)
            return [""] * n_samples, None

        prompt = _build_prompt(sample, context_level)
        task_id = str(sample.get("id", ""))
        entry_point = sample.get("entry_point", "")
        module = sample.get("module", "")
        target_fn_prompt = sample.get("target_function_prompt", "")
        gen = make_generate_fn(repo_root, task_id, entry_point, module, target_fn_prompt)
        completions = [_strip_markdown(gen(prompt)) for _ in range(n_samples)]
        return completions, traj_path

    generations: list[list[str]] = [[""] * n_samples for _ in range(len(dataset))]
    loop_traj_dirs: list[Optional[Path]] = [None] * len(dataset)

    if max_workers <= 1:
        for loop_idx, sample in enumerate(tqdm(dataset, desc=f"RepoExec/{context_level}")):
            try:
                generations[loop_idx], loop_traj_dirs[loop_idx] = _process_one(loop_idx, sample)
            except Exception as exc:
                logger.error("Task %d failed: %s", loop_idx, exc)
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_process_one, idx, sample): idx
                for idx, sample in enumerate(dataset)
            }
            for fut in tqdm(as_completed(futures), total=len(dataset),
                            desc=f"RepoExec/{context_level} (×{max_workers})"):
                idx = futures[fut]
                try:
                    generations[idx], loop_traj_dirs[idx] = fut.result()
                except Exception as exc:
                    logger.error("Task %d failed: %s", idx, exc)

    generations_file = output_dir / "generations.json"
    generations_file.write_text(json.dumps(generations))
    logger.info("Saved %d generations to %s", len(generations), generations_file)

    summary: dict = {"context_level": context_level, "n_samples": n_samples}

    exec_dir = repoexec_repo / "execution-code-eval"

    if (exec_dir / "execute.py").exists():
        # 2. process_result.py
        processed_file = output_dir / "processed_generations.json"
        proc_ok = _run_process_result(exec_dir, generations_file, output_dir, context_level, n_samples)
        if not proc_ok or not processed_file.exists():
            logger.warning("process_result.py failed or produced no output; skipping execution")
            for k in k_values:
                summary[f"pass@{k}"] = None
        else:
            # 3. execute.py (Docker per task)
            exec_results_dir = output_dir / "execution_results"
            exec_results_dir.mkdir(exist_ok=True)
            _run_execute(
                exec_dir, repoexec_repo, output_dir, exec_results_dir,
                context_level, dataset, max_workers=max_workers,
            )

            # 3b. annotate trajectory .md files with pass/fail results
            if trajectory_dir is not None:
                _annotate_trajectories(exec_results_dir, loop_traj_dirs, dataset)

            # 4. pass@k
            pass_at_k = _run_passk(
                exec_dir, exec_results_dir, n_samples, k_values,
                dataset=dataset, context_level=context_level,
            )
            summary.update(pass_at_k)
    else:
        logger.warning(
            "RepoExec execution-code-eval not found. "
            "Run scripts/setup_deps.sh and build the Docker image."
        )
        for k in k_values:
            summary[f"pass@{k}"] = None

    summary_file = output_dir / f"summary_{context_level}.json"
    summary_file.write_text(json.dumps(summary, indent=2))
    logger.info("RepoExec/%s results: %s", context_level, summary)
    return summary


def _build_prompt(sample: dict, context_level: str) -> str:
    return sample.get("prompt", sample.get("cross_context", ""))


def _strip_markdown(text: str) -> str:
    """Remove ```python / ``` fences and leading/trailing whitespace."""
    if "```python" in text:
        text = text[text.index("```python") + len("```python"):]
        text = text[:text.rfind("```")] if "```" in text else text
    elif "```" in text:
        text = text[text.index("```") + 3:]
        text = text[:text.rfind("```")] if "```" in text else text
    return text.strip()


def _run_process_result(
    exec_dir: Path,
    generations_file: Path,
    output_dir: Path,
    subset: str,
    n_samples: int,
) -> bool:
    cmd = [
        sys.executable,
        str(exec_dir / "process_result.py"),
        "--subset", subset,
        "--prediction_dir", str(output_dir.resolve()),
        "--n_samples", str(n_samples),
    ]
    result = subprocess.run(
        cmd,
        cwd=str(exec_dir),
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        logger.warning("process_result.py failed:\n%s\n%s", result.stdout, result.stderr)
        return False
    logger.info("process_result.py OK:\n%s", result.stdout[-500:])
    return True


def _run_execute(
    exec_dir: Path,
    repo_dir: Path,
    pred_dir: Path,
    save_dir: Path,
    subset: str,
    dataset: Any,
    max_workers: int = 1,
) -> None:
    pip_cache_vol = "repoexec_pip_cache_vol"
    pred_dir_abs = str(pred_dir.resolve())
    save_dir_abs = str(save_dir.resolve())
    repo_dir_abs = str(repo_dir.resolve())

    def _run_one(task_id: int) -> None:
        result_file = save_dir / f"results_{task_id}.jsonl"
        if result_file.exists():
            return
        project = dataset[task_id]["project"]  # e.g. "test-apps/python-string-utils"
        project_path = repo_dir_abs + "/" + project
        if not Path(project_path).exists():
            logger.warning("Project path not found, skipping task %d: %s", task_id, project_path)
            return

        cmd = (
            f"sudo docker run --rm "
            f"-v {pred_dir_abs}:/pred_dir:ro "
            f"-v {save_dir_abs}:/rs_dir "
            f"-v {repo_dir_abs}:/input:ro "
            f"-v {repo_dir_abs}/data_with_test_case:/output:ro "
            f"-v {project_path}/:/package:ro "
            f"-v {pip_cache_vol}:/tmp/pip_cache "
            f"-e PIP_CACHE_DIR=/tmp/pip_cache "
            f"codeeval-runner "
            f"--task_id {task_id} "
            f"--problem_file /pred_dir/processed_generations.json "
            f"--rs_dir /rs_dir "
            f"--timeout 120"
        )
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            log_file = save_dir / f"docker_error_{task_id}.log"
            log_file.write_text(result.stdout + result.stderr)
            logger.warning(
                "Docker execution failed for task %d (exit %d); log: %s",
                task_id, result.returncode, log_file,
            )

    task_ids = list(range(len(dataset)))
    if max_workers <= 1:
        for task_id in tqdm(task_ids, desc="Executing (Docker)"):
            _run_one(task_id)
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_run_one, tid) for tid in task_ids]
            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc=f"Executing (Docker ×{max_workers})"):
                try:
                    fut.result()
                except Exception as exc:
                    logger.error("Docker task failed: %s", exc)


def _estimate_pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator: 1 - C(n-c, k) / C(n, k)."""
    import math
    if n < k:
        return 1.0 if c > 0 else 0.0
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def _run_passk(
    exec_dir: Path,
    execution_dir: Path,
    n_samples: int,
    k_values: list[int],
    dataset: Any = None,
    context_level: str = "full_context",
) -> dict:
    """Compute pass@k directly from execution result files.

    Skips tasks where isContained=True in the dataset (same logic as the
    vendor passk.py, but using the correct dataset split).
    """
    result_files = sorted(
        execution_dir.glob("results_*.jsonl"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    if not result_files:
        logger.warning("No execution result files found in %s", execution_dir)
        return {f"pass@{k}": None for k in k_values}

    totals: list[int] = []
    corrects: list[int] = []
    skipped = 0

    for result_file in result_files:
        task_id = int(result_file.stem.split("_")[-1])

        # Skip "contained" tasks — the solution is visible in the given context
        if dataset is not None and task_id < len(dataset):
            if dataset[task_id].get("isContained", False):
                skipped += 1
                continue

        results = []
        for line in result_file.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        results = sorted(results, key=lambda r: r.get("prediction_id", 0))[:n_samples]
        if not results:
            continue

        passed = [r.get("passed", False) for r in results]
        totals.append(len(passed))
        corrects.append(sum(passed))

    if not totals:
        return {f"pass@{k}": None for k in k_values}

    logger.info(
        "pass@k: %d tasks evaluated, %d skipped (isContained)",
        len(totals), skipped,
    )

    pass_at_k = {}
    for k in k_values:
        if all(n >= k for n in totals):
            scores = [_estimate_pass_at_k(n, c, k) for n, c in zip(totals, corrects)]
            pass_at_k[f"pass@{k}"] = sum(scores) / len(scores)
        else:
            pass_at_k[f"pass@{k}"] = None

    result_json = execution_dir / "passk.json"
    result_json.write_text(json.dumps(pass_at_k, indent=2))
    logger.info("pass@k results: %s", pass_at_k)
    return pass_at_k


def _extract_failing_test(check: str, traceback: str) -> str:
    """
    Parse the failing test function name from a traceback and extract its
    source from the check program string.
    """
    import re as _re
    # Find the last "in test_XXX" frame in the traceback
    names = _re.findall(r"\bin (test_\w+)", traceback)
    if not names:
        return ""
    func_name = names[-1]
    # Extract the function body from check using AST
    try:
        import ast as _ast
        tree = _ast.parse(check)
        for node in tree.body:
            if isinstance(node, _ast.FunctionDef) and node.name == func_name:
                lines = check.splitlines(keepends=True)
                return "".join(lines[node.lineno - 1 : node.end_lineno])
    except Exception:
        pass
    # Fallback: regex extraction
    pattern = _re.compile(
        rf"^def {_re.escape(func_name)}\(.*?\n((?:[ \t]+.*\n?)*)", _re.MULTILINE
    )
    m = pattern.search(check)
    if m:
        return f"def {func_name}(...):\n{m.group(1)}"
    return ""


def _annotate_trajectories(
    exec_results_dir: Path,
    loop_traj_dirs: list,
    dataset: Any,
) -> None:
    """Append execution pass/fail results to trajectory .md files."""
    for loop_idx, traj_dir in enumerate(loop_traj_dirs):
        if traj_dir is None:
            continue
        result_file = exec_results_dir / f"results_{loop_idx}.jsonl"
        if not result_file.exists():
            continue

        results = []
        for line in result_file.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        if not results:
            continue

        passed = all(r.get("passed", False) for r in results)
        status = "PASSED" if passed else "FAILED"
        sample = dataset[loop_idx] if loop_idx < len(dataset) else {}
        check = sample.get("check", "")
        gold = sample.get("solution", "").strip()

        lines = ["", "---", f"## Execution Result: {status}", ""]
        for r in results:
            if not r.get("passed", False):
                msg = r.get("message", r.get("result", "")).strip()
                sample_idx = r.get("prediction_id", r.get("completion_id", "?"))
                lines += [f"**Sample {sample_idx} failure:**", "```", msg, "```", ""]
                test_src = _extract_failing_test(check, msg)
                if test_src:
                    lines += ["**Failing test:**", "```python", test_src.strip(), "```", ""]

        if not passed and gold:
            lines += ["**Gold solution:**", "```python", gold, "```", ""]

        section = "\n".join(lines)

        # Main .md has the same name as the trajectory directory
        md_file = traj_dir / f"{traj_dir.name}.md"
        if not md_file.exists():
            # Agent may produce turn files; pick the last one alphabetically
            md_files = sorted(traj_dir.glob("*.md"))
            if not md_files:
                logger.debug("No .md found in trajectory dir %s, skipping annotation", traj_dir)
                continue
            md_file = md_files[-1]

        try:
            md_file.write_text(md_file.read_text() + section)
        except Exception as exc:
            logger.warning("Failed to annotate %s: %s", md_file, exc)
