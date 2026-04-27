"""
Evaluation metrics: EM, ES, CodeBLEU, pass@k, DIR.
"""

from __future__ import annotations

import math
from difflib import SequenceMatcher
from typing import Callable

from codebleu import calc_codebleu


def compute_em(prediction: str, reference: str) -> float:
    return float(prediction.strip() == reference.strip())


def compute_es(prediction: str, reference: str) -> float:
    return SequenceMatcher(None, prediction.strip(), reference.strip()).ratio()


def compute_codebleu(prediction: str, reference: str, lang: str = "python") -> float:
    result = calc_codebleu([reference], [prediction], lang=lang)
    return float(result["codebleu"])


def compute_pass_at_k(n: int, c: int, k: int) -> float:
    """
    Unbiased pass@k estimator from the HumanEval paper.

    n: total samples generated
    c: number of correct samples
    k: k in pass@k
    """
    if n < k:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def compute_dir(predictions: list[str], execute_fn: Callable[[str], bool]) -> float:
    """
    Dependency-Inverted Ratio (DIR) from RepoExec.

    execute_fn(code) -> bool: returns True if the code passes tests.
    """
    if not predictions:
        return 0.0
    passed = sum(1 for pred in predictions if execute_fn(pred))
    return passed / len(predictions)


def aggregate_metrics(results: list[dict]) -> dict:
    if not results:
        return {}
    keys = [k for k in results[0] if isinstance(results[0][k], (int, float))]
    return {k: sum(r[k] for r in results) / len(results) for k in keys}
