"""Tests for evaluation metrics."""

import math

import pytest

from agentic_code_rag.eval.metrics import (
    aggregate_metrics,
    compute_dir,
    compute_em,
    compute_es,
    compute_pass_at_k,
)


def test_em_exact():
    assert compute_em("hello world", "hello world") == 1.0


def test_em_strip():
    assert compute_em("  hello  ", "hello") == 1.0


def test_em_mismatch():
    assert compute_em("hello", "world") == 0.0


def test_es_identical():
    assert compute_es("abc", "abc") == pytest.approx(1.0)


def test_es_empty():
    assert compute_es("", "") == pytest.approx(1.0)


def test_es_partial():
    score = compute_es("abcd", "abce")
    assert 0.0 < score < 1.0


def test_pass_at_k_all_correct():
    assert compute_pass_at_k(n=10, c=10, k=1) == pytest.approx(1.0)


def test_pass_at_k_none_correct():
    assert compute_pass_at_k(n=10, c=0, k=1) == pytest.approx(0.0)


def test_pass_at_k_partial():
    p = compute_pass_at_k(n=10, c=5, k=1)
    assert 0.0 < p < 1.0


def test_pass_at_k_k_greater_than_n():
    assert compute_pass_at_k(n=5, c=3, k=10) == 0.0


def test_pass_at_k_n_minus_c_less_than_k():
    assert compute_pass_at_k(n=5, c=4, k=2) == pytest.approx(1.0)


def test_dir_all_pass():
    result = compute_dir(["a", "b", "c"], execute_fn=lambda x: True)
    assert result == pytest.approx(1.0)


def test_dir_none_pass():
    result = compute_dir(["a", "b"], execute_fn=lambda x: False)
    assert result == pytest.approx(0.0)


def test_dir_empty():
    result = compute_dir([], execute_fn=lambda x: True)
    assert result == pytest.approx(0.0)


def test_aggregate_metrics_mean():
    results = [{"em": 1.0, "es": 0.8}, {"em": 0.0, "es": 0.4}]
    agg = aggregate_metrics(results)
    assert agg["em"] == pytest.approx(0.5)
    assert agg["es"] == pytest.approx(0.6)


def test_aggregate_metrics_empty():
    assert aggregate_metrics([]) == {}
