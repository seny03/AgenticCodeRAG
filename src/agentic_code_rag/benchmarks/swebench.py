"""
SWE-bench Verified runner.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

from tqdm import tqdm

from ._hf_utils import load_dataset_cached

logger = logging.getLogger(__name__)

SWEBENCH_VERIFIED = "SWE-bench/SWE-bench_Verified"


def run_swebench(
    agent_fn: Any,
    output_dir: Path,
    split: str = "test",
    max_instances: Optional[int] = None,
    run_harness: bool = False,
    predictions_id: str = "acr_agent",
) -> dict:
    """
    Run SWE-bench Verified evaluation.

    Parameters
    ----------
    agent_fn:
        Callable(instance: dict) -> str that returns a unified diff patch.
    output_dir:
        Where to write patches and results.
    split:
        Dataset split ("test").
    max_instances:
        Limit number of instances.
    run_harness:
        Whether to invoke the official Docker harness after generating patches.
    predictions_id:
        Identifier for this run (used by the harness).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading SWE-bench dataset: %s / %s", SWEBENCH_VERIFIED, split)
    instances = load_dataset_cached(SWEBENCH_VERIFIED, split=split)

    if max_instances:
        instances = instances.select(range(min(max_instances, len(instances))))

    predictions: list[dict] = []
    patches_dir = output_dir / "patches"
    patches_dir.mkdir(exist_ok=True)

    for instance in tqdm(instances, desc="SWE-bench/verified"):
        instance_id = instance["instance_id"]
        try:
            patch = agent_fn(instance)
        except Exception as exc:
            logger.warning("Agent failed on %s: %s", instance_id, exc)
            patch = ""

        pred = {
            "instance_id": instance_id,
            "model_patch": patch,
            "model_name_or_path": predictions_id,
        }
        predictions.append(pred)

        patch_file = patches_dir / f"{instance_id}.patch"
        patch_file.write_text(patch)

    predictions_file = output_dir / "predictions.jsonl"
    with open(predictions_file, "w") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")

    logger.info("Wrote %d predictions to %s", len(predictions), predictions_file)

    summary: dict = {
        "dataset": "verified",
        "n_instances": len(predictions),
        "n_with_patch": sum(1 for p in predictions if p["model_patch"].strip()),
    }

    if run_harness:
        harness_result = _run_official_harness(predictions_file, output_dir, predictions_id)
        summary.update(harness_result)

    summary_file = output_dir / "summary.json"
    summary_file.write_text(json.dumps(summary, indent=2))
    logger.info("SWE-bench summary: %s", summary)
    return summary


def _run_official_harness(
    predictions_file: Path,
    output_dir: Path,
    predictions_id: str,
) -> dict:
    """
    Invoke swebench.harness.run_evaluation.
    Runs with cwd=result_dir so logs/ and the report JSON land there.
    """
    result_dir = output_dir / "harness_results"
    result_dir.mkdir(exist_ok=True)

    cmd = [
        "sudo", "-E", sys.executable, "-m", "swebench.harness.run_evaluation",
        "--dataset_name", SWEBENCH_VERIFIED,
        "--predictions_path", str(predictions_file.resolve()),
        "--max_workers", "4",
        "--run_id", predictions_id,
    ]
    env = {**__import__("os").environ, "HF_DATASETS_OFFLINE": "1", "HF_HUB_OFFLINE": "1"}
    logger.info("Running SWE-bench harness: %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        cwd=str(result_dir),
        capture_output=True,
        text=True,
        timeout=7200,
        env=env,
    )
    if result.returncode != 0:
        logger.warning("SWE-bench harness failed:\n%s", result.stderr)
        return {"harness_error": result.stderr[:500]}

    # Report file: {model_name_or_path}.{run_id}.json written to cwd
    report_file = result_dir / f"acr_agent.{predictions_id}.json"
    if not report_file.exists():
        # fallback: any *.json in result_dir
        candidates = list(result_dir.glob(f"*.{predictions_id}.json"))
        report_file = candidates[0] if candidates else None

    if report_file and report_file.exists():
        data = json.loads(report_file.read_text())
        resolved = data.get("resolved_ids", data.get("resolved", []))
        total = data.get("total_instances", data.get("total", len(resolved)))
        return {
            "resolved_rate": len(resolved) / total if total else 0.0,
            "resolved_count": len(resolved),
            "total": total,
        }
    return {}
