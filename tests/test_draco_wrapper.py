"""Tests for DraCoIndex and DraCoToolAPI."""

import json
import tempfile
from pathlib import Path

import pytest

from agentic_code_rag.draco_wrapper.index import DraCoIndex, GraphNode, GraphEdge
from agentic_code_rag.draco_wrapper.tools import DraCoToolAPI


@pytest.fixture
def sample_graph_json(tmp_path: Path) -> Path:
    data = {
        "nodes": [
            {"node_id": "mod.foo", "kind": "function", "name": "foo", "file_path": "src/mod.py", "start_line": 10, "end_line": 20, "metadata": {}},
            {"node_id": "mod.bar", "kind": "function", "name": "bar", "file_path": "src/mod.py", "start_line": 25, "end_line": 35, "metadata": {}},
            {"node_id": "mod.Baz", "kind": "class",    "name": "Baz", "file_path": "src/mod.py", "start_line": 40, "end_line": 60, "metadata": {}},
        ],
        "edges": [
            {"source": "mod.bar", "target": "mod.foo", "edge_type": "calls", "metadata": {}},
            {"source": "mod.Baz", "target": "mod.foo", "edge_type": "calls", "metadata": {}},
        ],
    }
    graph_dir = tmp_path / ".draco_graph"
    graph_dir.mkdir()
    (graph_dir / "graph.json").write_text(json.dumps(data))
    return tmp_path


@pytest.fixture
def index(sample_graph_json: Path) -> DraCoIndex:
    idx = DraCoIndex(sample_graph_json)
    idx._load_graph()
    return idx


@pytest.fixture
def repo_with_files(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    src.mkdir()
    (src / "mod.py").write_text(
        "def foo():\n    pass\n\ndef bar():\n    foo()\n\nclass Baz:\n    def method(self):\n        foo()\n"
    )
    return tmp_path


def test_index_loads_nodes(index: DraCoIndex) -> None:
    assert index.node_count == 3
    assert index.edge_count == 2


def test_search_symbol_exact(index: DraCoIndex) -> None:
    results = index.search_symbol("foo")
    assert len(results) == 1
    assert results[0].name == "foo"


def test_search_symbol_partial(index: DraCoIndex) -> None:
    results = index.search_symbol("ba")
    names = {r.name for r in results}
    assert "bar" in names or "Baz" in names


def test_search_symbol_kind_filter(index: DraCoIndex) -> None:
    results = index.search_symbol("", kind="class")
    assert all(r.kind == "class" for r in results)


def test_get_callers(index: DraCoIndex) -> None:
    callers = index.get_callers("mod.foo")
    caller_sources = {e.source for e in callers}
    assert "mod.bar" in caller_sources
    assert "mod.Baz" in caller_sources


def test_get_callees(index: DraCoIndex) -> None:
    callees = index.get_callees("mod.bar")
    assert any(e.target == "mod.foo" for e in callees)


def test_get_neighbors_hops(index: DraCoIndex) -> None:
    neighbors = index.get_neighbors("mod.foo", hops=1)
    assert len(neighbors) > 0


def test_file_index(index: DraCoIndex) -> None:
    nodes = index.get_file_nodes("src/mod.py")
    assert len(nodes) == 3


def test_save_and_reload_json(index: DraCoIndex, tmp_path: Path) -> None:
    out = tmp_path / "graph_out.json"
    index.save_json(out)
    idx2 = DraCoIndex(tmp_path)
    idx2._load_from_json(out)
    assert idx2.node_count == index.node_count
    assert idx2.edge_count == index.edge_count


def test_tool_repo_tree(repo_with_files: Path, index: DraCoIndex) -> None:
    tool = DraCoToolAPI(index, repo_with_files)
    tree = tool.repo_tree(".", depth=2)
    assert "src" in tree
    assert "mod.py" in tree


def test_tool_open_file(repo_with_files: Path, index: DraCoIndex) -> None:
    tool = DraCoToolAPI(index, repo_with_files)
    content = tool.open_file("src/mod.py", start_line=1, n_lines=3)
    assert "def foo" in content
    assert "1 |" in content


def test_tool_open_file_not_found(repo_with_files: Path, index: DraCoIndex) -> None:
    tool = DraCoToolAPI(index, repo_with_files)
    result = tool.open_file("nonexistent.py")
    assert "not found" in result.lower()


def test_tool_grep_search(repo_with_files: Path, index: DraCoIndex) -> None:
    tool = DraCoToolAPI(index, repo_with_files)
    results = tool.grep_search("def foo", glob="**/*.py")
    assert "mod.py" in results


def test_tool_grep_search_no_match(repo_with_files: Path, index: DraCoIndex) -> None:
    tool = DraCoToolAPI(index, repo_with_files)
    results = tool.grep_search("xyzzy_not_found")
    assert "No matches" in results


def test_tool_symbol_search(repo_with_files: Path, index: DraCoIndex) -> None:
    tool = DraCoToolAPI(index, repo_with_files)
    result = tool.symbol_search("foo")
    assert "foo" in result


def test_tool_context_pack(repo_with_files: Path, index: DraCoIndex) -> None:
    tool = DraCoToolAPI(index, repo_with_files)
    result = tool.context_pack("foo", token_budget=500)
    assert "foo" in result or "Definition" in result
