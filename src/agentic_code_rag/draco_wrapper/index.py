"""
Wrapper around DraCo's repo-specific context graph.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

DRACO_DIR_ENV = "DRACO_DIR"
DEFAULT_DRACO_DIR = Path(__file__).resolve().parents[3] / "vendor" / "DraCo"


@dataclass
class GraphNode:
    node_id: str
    kind: str          # "function", "class", "variable", "file", "import", ...
    name: str
    file_path: str
    start_line: int = 0
    end_line: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass
class GraphEdge:
    source: str
    target: str
    edge_type: str     # "calls", "called_by", "imports", "defines", "uses", ...
    metadata: dict = field(default_factory=dict)


class DraCoIndex:
    """
    Interface to a DraCo repo-specific context graph.

    Wraps the graph built by DraCo's preprocess.py and provides
    query methods for the agent.
    """

    def __init__(
        self,
        repo_root: Path,
        graph_dir: Optional[Path] = None,
        draco_dir: Optional[Path] = None,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        _env_draco = os.environ.get(DRACO_DIR_ENV, "").strip()
        self.draco_dir = Path(draco_dir or _env_draco or DEFAULT_DRACO_DIR)
        self.graph_dir = Path(graph_dir) if graph_dir else (self.repo_root / ".draco_graph")

        self._nodes: dict[str, GraphNode] = {}
        self._edges: list[GraphEdge] = []
        self._adj: dict[str, list[GraphEdge]] = {}        # node_id -> outgoing edges
        self._rev_adj: dict[str, list[GraphEdge]] = {}    # node_id -> incoming edges
        self._file_index: dict[str, list[str]] = {}       # file_path -> [node_ids]
        self._name_index: dict[str, list[str]] = {}       # name -> [node_ids]

    def build(self, force: bool = False) -> None:
        """Build a DraCo-style context graph for *repo_root*."""
        if self.graph_dir.exists() and not force:
            marker = self.graph_dir / "graph.json"
            if marker.exists():
                logger.info("Graph already exists at %s, loading", self.graph_dir)
                self._load_graph()
                return

        self.graph_dir.mkdir(parents=True, exist_ok=True)
        draco_src = self.draco_dir / "src"

        if not (draco_src / "preprocess.py").exists():
            raise FileNotFoundError(
                f"DraCo src/ not found at {draco_src}. "
                f"Clone DraCo to {self.draco_dir} or set {DRACO_DIR_ENV} env var."
            )

        logger.info("Building DraCo graph for %s", self.repo_root)

        draco_src_str = str(draco_src)
        _added = draco_src_str not in sys.path
        if _added:
            sys.path.insert(0, draco_src_str)
        try:
            from preprocess import projectParser  # noqa: PLC0415
            parse_result: dict = projectParser().parse_dir(str(self.repo_root))
        finally:
            if _added and draco_src_str in sys.path:
                sys.path.remove(draco_src_str)

        graph_data = self._draco_parse_to_graph(parse_result)
        out = self.graph_dir / "graph.json"
        out.write_text(json.dumps(graph_data, indent=2))
        logger.info("DraCo graph saved: %d nodes, %d edges",
                    len(graph_data["nodes"]), len(graph_data["edges"]))
        self._load_graph()

    def _draco_parse_to_graph(self, parse_result: dict) -> dict:
        nodes: list[dict] = []
        edges: list[dict] = []
        seen_ids: set[str] = set()

        def _id(module: str, name: str) -> str:
            return f"{module}.{name}" if name else module

        type_map = {"Function": "function", "Class": "class",
                    "Variable": "variable", "Module": "module"}
        rel_type_map = {"Assign": "uses", "Hint": "uses",
                        "Rhint": "calls", "Inherit": "extends"}

        for module, symbols in parse_result.items():
            for name, info in symbols.items():
                node_id = _id(module, name)
                if node_id not in seen_ids:
                    seen_ids.add(node_id)
                    nodes.append({
                        "node_id": node_id,
                        "kind": type_map.get(info.get("type", ""), "unknown"),
                        "name": name,
                        "file_path": module.replace(".", "/") + ".py",
                        "start_line": info.get("sline", 0),
                        "end_line": info.get("sline", 0),
                        "metadata": {},
                    })
                for rel in info.get("rels", []):
                    if len(rel) < 3:
                        continue
                    edges.append({
                        "source": node_id,
                        "target": _id(module, rel[0]),
                        "edge_type": rel_type_map.get(rel[2], rel[2].lower()),
                        "metadata": {},
                    })
                imp = info.get("import")
                if imp:
                    edges.append({
                        "source": node_id,
                        "target": _id(imp[0], imp[1] or ""),
                        "edge_type": "imports",
                        "metadata": {},
                    })

        return {"nodes": nodes, "edges": edges}

    def _load_graph(self) -> None:
        """Load graph from the graph_dir artifacts."""
        graph_file = self.graph_dir / "graph.json"
        if graph_file.exists():
            self._load_from_json(graph_file)
            return

        self._load_from_draco_native()

    def _load_from_json(self, path: Path) -> None:
        data = json.loads(path.read_text())
        for n in data.get("nodes", []):
            node = GraphNode(
                node_id=n["node_id"],
                kind=n.get("kind", "unknown"),
                name=n.get("name", ""),
                file_path=n.get("file_path", ""),
                start_line=n.get("start_line", 0),
                end_line=n.get("end_line", 0),
                metadata=n.get("metadata", {}),
            )
            self._nodes[node.node_id] = node
        for e in data.get("edges", []):
            edge = GraphEdge(
                source=e["source"],
                target=e["target"],
                edge_type=e.get("edge_type", "unknown"),
                metadata=e.get("metadata", {}),
            )
            self._edges.append(edge)
        self._build_indexes()
        logger.info("Loaded graph: %d nodes, %d edges", len(self._nodes), len(self._edges))

    def _load_from_draco_native(self) -> None:
        """
        Attempt to load DraCo's native pickle/json artifacts.
        """
        import pickle

        for pkl_name in ["context_graph.pkl", "repo_graph.pkl", "graph.pkl"]:
            pkl_path = self.graph_dir / pkl_name
            if pkl_path.exists():
                with open(pkl_path, "rb") as f:
                    data = pickle.load(f)
                self._convert_draco_native(data)
                return

        logger.warning("No graph artifacts found in %s", self.graph_dir)

    def _convert_draco_native(self, data: Any) -> None:
        """Convert DraCo's native graph format to our internal representation."""
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(key, str):
                    if key not in self._nodes:
                        self._nodes[key] = GraphNode(
                            node_id=key, kind="unknown", name=key,
                            file_path="", start_line=0, end_line=0,
                        )
                    if isinstance(value, (list, set)):
                        for target in value:
                            target_str = str(target)
                            if target_str not in self._nodes:
                                self._nodes[target_str] = GraphNode(
                                    node_id=target_str, kind="unknown", name=target_str,
                                    file_path="", start_line=0, end_line=0,
                                )
                            self._edges.append(GraphEdge(
                                source=key, target=target_str, edge_type="related",
                            ))
        self._build_indexes()
        logger.info("Converted DraCo native graph: %d nodes, %d edges",
                     len(self._nodes), len(self._edges))

    def _build_indexes(self) -> None:
        self._adj.clear()
        self._rev_adj.clear()
        self._file_index.clear()
        self._name_index.clear()

        for edge in self._edges:
            self._adj.setdefault(edge.source, []).append(edge)
            self._rev_adj.setdefault(edge.target, []).append(edge)

        for node in self._nodes.values():
            if node.file_path:
                self._file_index.setdefault(node.file_path, []).append(node.node_id)
            if node.name:
                key = node.name.lower()
                self._name_index.setdefault(key, []).append(node.node_id)


    def search_symbol(self, query: str, kind: Optional[str] = None, top_k: int = 20) -> list[GraphNode]:
        """Fuzzy search over symbol names."""
        query_lower = query.lower()
        results: list[tuple[float, GraphNode]] = []
        for node in self._nodes.values():
            if kind and node.kind != kind:
                continue
            name_lower = node.name.lower()
            if query_lower in name_lower:
                score = 1.0 if name_lower.startswith(query_lower) else 0.5
                if name_lower == query_lower:
                    score = 2.0
                results.append((score, node))
        results.sort(key=lambda x: x[0], reverse=True)
        return [n for _, n in results[:top_k]]

    def search_file(self, query: str, top_k: int = 20) -> list[str]:
        """Search file paths by substring match."""
        query_lower = query.lower()
        matches: list[tuple[float, str]] = []
        seen: set[str] = set()
        for node in self._nodes.values():
            if node.file_path and node.file_path not in seen:
                seen.add(node.file_path)
                fp_lower = node.file_path.lower()
                if query_lower in fp_lower:
                    score = 1.0 if fp_lower.endswith(query_lower) else 0.5
                    matches.append((score, node.file_path))
        matches.sort(key=lambda x: x[0], reverse=True)
        return [fp for _, fp in matches[:top_k]]

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        return self._nodes.get(node_id)

    def get_outgoing(self, node_id: str, edge_types: Optional[list[str]] = None) -> list[GraphEdge]:
        edges = self._adj.get(node_id, [])
        if edge_types:
            edges = [e for e in edges if e.edge_type in edge_types]
        return edges

    def get_incoming(self, node_id: str, edge_types: Optional[list[str]] = None) -> list[GraphEdge]:
        edges = self._rev_adj.get(node_id, [])
        if edge_types:
            edges = [e for e in edges if e.edge_type in edge_types]
        return edges

    def get_neighbors(
        self,
        node_id: str,
        edge_types: Optional[list[str]] = None,
        hops: int = 1,
    ) -> list[dict]:
        """BFS over the graph up to *hops* levels."""
        visited: set[str] = {node_id}
        frontier = [node_id]
        results: list[dict] = []

        for depth in range(hops):
            next_frontier: list[str] = []
            for nid in frontier:
                for edge in self.get_outgoing(nid, edge_types):
                    if edge.target not in visited:
                        visited.add(edge.target)
                        next_frontier.append(edge.target)
                        results.append({
                            "from": nid,
                            "to": edge.target,
                            "edge_type": edge.edge_type,
                            "depth": depth + 1,
                        })
                for edge in self.get_incoming(nid, edge_types):
                    if edge.source not in visited:
                        visited.add(edge.source)
                        next_frontier.append(edge.source)
                        results.append({
                            "from": edge.source,
                            "to": nid,
                            "edge_type": edge.edge_type,
                            "depth": depth + 1,
                        })
            frontier = next_frontier
        return results

    def get_file_nodes(self, file_path: str) -> list[GraphNode]:
        node_ids = self._file_index.get(file_path, [])
        return [self._nodes[nid] for nid in node_ids if nid in self._nodes]

    def get_callers(self, node_id: str, top_k: int = 20) -> list[GraphEdge]:
        edges = self.get_incoming(node_id, edge_types=["calls", "called_by", "invokes"])
        return edges[:top_k]

    def get_callees(self, node_id: str, top_k: int = 20) -> list[GraphEdge]:
        edges = self.get_outgoing(node_id, edge_types=["calls", "called_by", "invokes"])
        return edges[:top_k]

    def save_json(self, path: Optional[Path] = None) -> None:
        """Serialize the graph to JSON for portability."""
        path = path or (self.graph_dir / "graph.json")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "nodes": [
                {
                    "node_id": n.node_id,
                    "kind": n.kind,
                    "name": n.name,
                    "file_path": n.file_path,
                    "start_line": n.start_line,
                    "end_line": n.end_line,
                    "metadata": n.metadata,
                }
                for n in self._nodes.values()
            ],
            "edges": [
                {
                    "source": e.source,
                    "target": e.target,
                    "edge_type": e.edge_type,
                    "metadata": e.metadata,
                }
                for e in self._edges
            ],
        }
        Path(path).write_text(json.dumps(data, indent=2))
        logger.info("Graph saved to %s", path)

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return len(self._edges)
