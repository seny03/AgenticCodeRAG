"""
Tool API layer on top of DraCoIndex.

Each method here corresponds to one tool the LangGraph agent can call.
The methods return plain dicts/strings suitable for LLM consumption.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from .index import DraCoIndex

logger = logging.getLogger(__name__)


class DraCoToolAPI:
    """
    Thin wrapper that turns DraCoIndex queries into agent-friendly tool outputs.
    Also provides file-level tools (open_file, grep, repo_tree, etc.)
    """

    def __init__(self, index: DraCoIndex, repo_root: Path) -> None:
        self.index = index
        self.repo_root = Path(repo_root).resolve()

    def repo_tree(self, path: str = ".", depth: int = 3) -> str:
        """Return a tree-like listing of the repo directory."""
        target = self.repo_root / path
        if not target.exists():
            return f"Path not found: {path}"
        lines: list[str] = []
        self._walk_tree(target, "", depth, lines)
        return "\n".join(lines[:500])

    def _walk_tree(self, root: Path, prefix: str, depth: int, lines: list[str]) -> None:
        if depth < 0:
            return
        try:
            entries = sorted(root.iterdir(), key=lambda p: (not p.is_dir(), p.name))
        except PermissionError:
            return
        for i, entry in enumerate(entries):
            if entry.name.startswith("."):
                continue
            is_last = i == len(entries) - 1
            connector = "└── " if is_last else "├── "
            lines.append(f"{prefix}{connector}{entry.name}")
            if entry.is_dir():
                extension = "    " if is_last else "│   "
                self._walk_tree(entry, prefix + extension, depth - 1, lines)

    def open_file(self, path: str, start_line: int = 1, n_lines: int = 100) -> str:
        """Read a slice of a file."""
        full = self.repo_root / path
        if not full.exists() and path.endswith(".py"):
            # module.py may actually be a package: module/__init__.py
            init = self.repo_root / path[:-3] / "__init__.py"
            if init.exists():
                full = init
                path = str(init.relative_to(self.repo_root))
        if not full.exists():
            return f"File not found: {path}"
        try:
            all_lines = full.read_text(errors="replace").splitlines()
        except Exception as exc:
            return f"Error reading {path}: {exc}"
        start = max(0, start_line - 1)
        end = start + n_lines
        selected = all_lines[start:end]
        numbered = [f"{start + i + 1:>6} | {line}" for i, line in enumerate(selected)]
        header = f"# {path}  (lines {start+1}-{min(end, len(all_lines))} of {len(all_lines)})"
        return header + "\n" + "\n".join(numbered)

    def grep_search(self, query: str, glob: str = "**/*", top_k: int = 20) -> str:
        """Regex search across files."""
        try:
            pattern = re.compile(query, re.IGNORECASE)
        except re.error:
            pattern = re.compile(re.escape(query), re.IGNORECASE)
        results: list[str] = []
        for p in self.repo_root.glob(glob):
            if not p.is_file() or p.stat().st_size > 1_000_000:
                continue
            try:
                for i, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
                    if pattern.search(line):
                        rel = str(p.relative_to(self.repo_root))
                        results.append(f"{rel}:{i}: {line.strip()}")
                        if len(results) >= top_k:
                            break
            except Exception:
                continue
            if len(results) >= top_k:
                break
        if not results:
            return f"No matches for '{query}'"
        return "\n".join(results)

    def symbol_search(self, query: str, kind: Optional[str] = None, top_k: int = 20) -> str:
        """Search symbols in the DraCo graph."""
        nodes = self.index.search_symbol(query, kind=kind, top_k=top_k)
        if not nodes:
            return f"No symbols matching '{query}'"
        parts = [f"Found {len(nodes)} symbols:"]
        for n in nodes:
            loc = f"{n.file_path}:{n.start_line}" if n.file_path else "?"
            parts.append(f"  {n.kind:<12} {n.name:<40} {loc}")
            if n.file_path and n.start_line:
                for line in self._extract_snippet(n.file_path, n.start_line, n.end_line):
                    parts.append(f"    {line}")
        return "\n".join(parts)

    def goto_definition(self, symbol_or_location: str) -> str:
        """Jump to the definition of a symbol."""
        nodes = self.index.search_symbol(symbol_or_location, top_k=5)
        if not nodes:
            return f"Symbol not found: {symbol_or_location}"
        node = nodes[0]
        if not node.file_path:
            return f"No file location for symbol: {node.name}"
        return self.open_file(node.file_path, start_line=max(1, node.start_line), n_lines=30)

    def find_references(self, symbol: str, top_k: int = 20) -> str:
        """Find all references to a symbol via the graph."""
        nodes = self.index.search_symbol(symbol, top_k=1)
        if not nodes:
            return f"Symbol not found: {symbol}"
        node = nodes[0]
        incoming = self.index.get_incoming(node.node_id)
        if not incoming:
            return f"No references found for '{symbol}'"
        parts = [f"References to '{symbol}' ({len(incoming[:top_k])}):"]
        for edge in incoming[:top_k]:
            src_node = self.index.get_node(edge.source)
            if src_node:
                loc = f"{src_node.file_path}:{src_node.start_line}" if src_node.file_path else "?"
                parts.append(f"  [{edge.edge_type}] {src_node.name} at {loc}")
                if src_node.file_path and src_node.start_line:
                    for line in self._extract_snippet(src_node.file_path, src_node.start_line, src_node.end_line):
                        parts.append(f"    {line}")
            else:
                parts.append(f"  [{edge.edge_type}] {edge.source}")
        return "\n".join(parts)

    def find_implementations(self, symbol: str, top_k: int = 20) -> str:
        """Find implementations/overrides of a symbol."""
        nodes = self.index.search_symbol(symbol, top_k=1)
        if not nodes:
            return f"Symbol not found: {symbol}"
        node = nodes[0]
        incoming = self.index.get_incoming(
            node.node_id, edge_types=["implements", "overrides", "extends"]
        )
        if not incoming:
            return f"No implementations found for '{symbol}'"
        parts = [f"Implementations of '{symbol}' ({len(incoming[:top_k])}):"]
        for edge in incoming[:top_k]:
            src_node = self.index.get_node(edge.source)
            if src_node:
                loc = f"{src_node.file_path}:{src_node.start_line}" if src_node.file_path else "?"
                parts.append(f"  {src_node.name} at {loc}")
                if src_node.file_path and src_node.start_line:
                    for line in self._extract_snippet(src_node.file_path, src_node.start_line, src_node.end_line):
                        parts.append(f"    {line}")
            else:
                parts.append(f"  {edge.source}")
        return "\n".join(parts)

    def get_callers(self, symbol: str, top_k: int = 20) -> str:
        """Find callers of a symbol."""
        nodes = self.index.search_symbol(symbol, top_k=1)
        if not nodes:
            return f"Symbol not found: {symbol}"
        edges = self.index.get_callers(nodes[0].node_id, top_k=top_k)
        if not edges:
            return f"No callers found for '{symbol}'"
        parts = [f"Callers of '{symbol}' ({len(edges)}):"]
        for edge in edges:
            src_node = self.index.get_node(edge.source)
            if src_node:
                loc = f"{src_node.file_path}:{src_node.start_line}" if src_node.file_path else "?"
                parts.append(f"  {src_node.name} at {loc}")
                if src_node.file_path and src_node.start_line:
                    for line in self._extract_snippet(src_node.file_path, src_node.start_line, src_node.end_line):
                        parts.append(f"    {line}")
            else:
                parts.append(f"  {edge.source}")
        return "\n".join(parts)

    def get_callees(self, symbol: str, top_k: int = 20) -> str:
        """Find callees of a symbol."""
        nodes = self.index.search_symbol(symbol, top_k=1)
        if not nodes:
            return f"Symbol not found: {symbol}"
        edges = self.index.get_callees(nodes[0].node_id, top_k=top_k)
        if not edges:
            return f"No callees found for '{symbol}'"
        parts = [f"Callees of '{symbol}' ({len(edges)}):"]
        for edge in edges:
            tgt_node = self.index.get_node(edge.target)
            if tgt_node:
                loc = f"{tgt_node.file_path}:{tgt_node.start_line}" if tgt_node.file_path else "?"
                parts.append(f"  {tgt_node.name} at {loc}")
                if tgt_node.file_path and tgt_node.start_line:
                    for line in self._extract_snippet(tgt_node.file_path, tgt_node.start_line, tgt_node.end_line):
                        parts.append(f"    {line}")
            else:
                parts.append(f"  {edge.target}")
        return "\n".join(parts)

    def graph_neighbors(
        self, node_id: str, edge_types: Optional[list[str]] = None, hops: int = 1
    ) -> str:
        """BFS over the graph from a node."""
        results = self.index.get_neighbors(node_id, edge_types=edge_types, hops=hops)
        if not results:
            return f"No neighbors found for '{node_id}'"
        lines = []
        for r in results:
            lines.append(f"  {r['from']} --[{r['edge_type']}]--> {r['to']}  (depth {r['depth']})")
        return f"Graph neighbors ({len(lines)}):\n" + "\n".join(lines)

    def retrieve_context_for_location(
        self, file_path: str, line: int, token_budget: int = 4000
    ) -> str:
        """
        Retrieve structured context around a specific location.
        Gathers: the target code, nearby symbols, imports, related tests.
        """
        parts: list[str] = []
        budget_used = 0

        code = self.open_file(file_path, start_line=max(1, line - 10), n_lines=40)
        parts.append("=== Target Code ===\n" + code)
        budget_used += len(code.split())

        file_nodes = self.index.get_file_nodes(file_path)
        if file_nodes:
            sym_lines = []
            for n in file_nodes[:10]:
                sym_lines.append(f"  {n.kind}: {n.name} (line {n.start_line})")
            parts.append("=== File Symbols ===\n" + "\n".join(sym_lines))

        target_node = None
        for n in file_nodes:
            if n.start_line <= line <= n.end_line:
                target_node = n
                break

        if target_node:
            neighbors = self.index.get_neighbors(target_node.node_id, hops=1)
            if neighbors:
                nb_lines = []
                for nb in neighbors[:10]:
                    nb_lines.append(f"  {nb['from']} --[{nb['edge_type']}]--> {nb['to']}")
                parts.append("=== Related Symbols ===\n" + "\n".join(nb_lines))

        return "\n\n".join(parts)

    def retrieve_context_for_task(self, task_text: str, token_budget: int = 4000) -> str:
        """
        Given a task description, retrieve relevant context from the graph.
        Extracts keywords, searches symbols and files, gathers context.
        """
        words = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", task_text)
        keywords = [w for w in words if len(w) > 2][:10]

        parts: list[str] = []
        seen_files: set[str] = set()

        for kw in keywords:
            nodes = self.index.search_symbol(kw, top_k=3)
            for node in nodes:
                if node.file_path and node.file_path not in seen_files:
                    seen_files.add(node.file_path)
                    code = self.open_file(
                        node.file_path,
                        start_line=max(1, node.start_line - 2),
                        n_lines=20,
                    )
                    parts.append(f"=== {node.name} ({node.kind}) ===\n{code}")
                    if len(parts) >= 8:
                        break
            if len(parts) >= 8:
                break

        if not parts:
            return "No relevant context found for the given task."
        return "\n\n".join(parts)

    def apply_patch(self, unified_diff: str) -> str:
        """Apply a unified diff to the repo."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False) as f:
            f.write(unified_diff)
            patch_path = f.name
        try:
            result = subprocess.run(
                ["patch", "-p1", "--forward", "-i", patch_path],
                cwd=str(self.repo_root),
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                return "Patch applied successfully.\n" + result.stdout
            return f"Patch failed (exit {result.returncode}):\n{result.stderr}\n{result.stdout}"
        finally:
            os.unlink(patch_path)

    def run_lint(self, paths: Optional[list[str]] = None) -> str:
        """Run linting on specified paths."""
        targets = paths or ["."]
        cmd = ["python", "-m", "ruff", "check"] + targets
        result = subprocess.run(
            cmd,
            cwd=str(self.repo_root),
            capture_output=True,
            text=True,
            timeout=120,
        )
        output = result.stdout + result.stderr
        return output[:5000] if output else "Lint passed (no output)."

    def run_tests(self, selector: str = "") -> str:
        """Run tests with an optional selector."""
        cmd = ["python", "-m", "pytest", "-x", "--tb=short"]
        if selector:
            cmd.append(selector)
        result = subprocess.run(
            cmd,
            cwd=str(self.repo_root),
            capture_output=True,
            text=True,
            timeout=300,
        )
        output = result.stdout + result.stderr
        return output[:8000] if output else "Tests completed (no output)."

    def run_command(self, cmd: str) -> str:
        """Run an arbitrary command."""
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=str(self.repo_root),
            capture_output=True,
            text=True,
            timeout=120,
        )
        output = result.stdout + result.stderr
        return output[:5000] if output else f"Command exited with code {result.returncode}."

    def show_diff(self) -> str:
        """Show the current uncommitted diff in the repo."""
        result = subprocess.run(
            ["git", "diff"],
            cwd=str(self.repo_root),
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            result = subprocess.run(
                ["diff", "-ru", ".", "."],
                cwd=str(self.repo_root),
                capture_output=True,
                text=True,
                timeout=30,
            )
        return result.stdout[:8000] if result.stdout else "No changes detected."

    def bm25_search(self, query: str, top_k: int = 20) -> str:
        """
        BM25 full-text search over all source files in the repo.

        Uses rank-bm25. The corpus is built lazily on first call and cached.
        Returns ranked hits with short code snippets.
        """
        if not hasattr(self, "_bm25_index"):
            self._bm25_index = self._build_bm25()
        hits = self._bm25_index.search(query, top_k=top_k)
        if not hits:
            return f"No BM25 results for '{query}'"

        parts: list[str] = [f"BM25 results ({len(hits)}):"]
        for h in hits:
            header = f"  [{h.kind}] {h.doc_id}  score={h.score:.3f}"
            snippet = self._extract_snippet(h.file_path, h.start_line, h.end_line)
            if snippet:
                parts.append(header)
                for sline in snippet:
                    parts.append(f"    {sline}")
            else:
                parts.append(header)
        return "\n".join(parts)

    def _extract_snippet(
        self, file_path: str, start_line: int, end_line: int, max_lines: int = 6
    ) -> list[str]:
        """Read up to max_lines from file_path starting at start_line."""
        if not file_path:
            return []
        full = self.repo_root / file_path
        if not full.exists():
            return []
        try:
            all_lines = full.read_text(errors="replace").splitlines()
        except Exception:
            return []
        s = max(0, start_line - 1)
        e = min(len(all_lines), end_line if end_line > s else s + max_lines)
        e = min(e, s + max_lines)
        return [f"{s + i + 1:>4} | {line}" for i, line in enumerate(all_lines[s:e])]

    def _build_bm25(self):
        from rank_bm25 import BM25Okapi
        import re

        corpus_meta: list[dict] = []
        tokenized: list[list[str]] = []

        for node in self.index._nodes.values():
            if not node.file_path:
                continue
            # include source lines in BM25 text if available
            source_text = ""
            if node.start_line > 0:
                lines = self._extract_snippet(node.file_path, node.start_line, node.end_line, max_lines=20)
                source_text = " ".join(l.split("|", 1)[-1] for l in lines)
            text = f"{node.name} {node.kind} {node.file_path} {source_text}"
            tokens = re.findall(r"[a-zA-Z0-9_]+", text.lower())
            corpus_meta.append({
                "doc_id": node.node_id,
                "kind": "symbol",
                "file_path": node.file_path,
                "start_line": node.start_line,
                "end_line": node.end_line,
            })
            tokenized.append(tokens)

        for p in self.repo_root.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix not in {".py", ".ts", ".js", ".java", ".go", ".rs", ".cpp", ".c"}:
                continue
            if p.stat().st_size > 500_000:
                continue
            try:
                content = p.read_text(errors="replace")
                tokens = re.findall(r"[a-zA-Z0-9_]+", content.lower())
                rel = str(p.relative_to(self.repo_root))
                corpus_meta.append({
                    "doc_id": rel,
                    "kind": "file",
                    "file_path": rel,
                    "start_line": 1,
                    "end_line": 6,
                })
                tokenized.append(tokens)
            except Exception:
                continue

        bm25 = BM25Okapi(tokenized)

        class _BM25Index:
            def search(self, query: str, top_k: int = 20):
                import re
                from dataclasses import dataclass

                @dataclass
                class Hit:
                    doc_id: str
                    kind: str
                    score: float
                    file_path: str
                    start_line: int
                    end_line: int

                q_tokens = re.findall(r"[a-zA-Z0-9_]+", query.lower())
                scores = bm25.get_scores(q_tokens)
                ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
                results = []
                for idx, score in ranked[:top_k]:
                    if score <= 0:
                        break
                    m = corpus_meta[idx]
                    results.append(Hit(
                        doc_id=m["doc_id"], kind=m["kind"], score=float(score),
                        file_path=m["file_path"], start_line=m["start_line"], end_line=m["end_line"],
                    ))
                return results

        return _BM25Index()

    def vector_search(self, query: str, top_k: int = 20) -> str:
        """
        Dense vector search over symbol names and file paths.

        Uses sentence-transformers + faiss. Index is built lazily on first call.
        """
        if not hasattr(self, "_vector_index"):
            self._vector_index = self._build_vector_index()
        if self._vector_index is None:
            return "Vector index not available (sentence-transformers or faiss not installed)."
        hits = self._vector_index.search(query, top_k=top_k)
        if not hits:
            return f"No vector results for '{query}'"
        parts: list[str] = [f"Vector results ({len(hits)}):"]
        for h in hits:
            parts.append(f"  [{h['kind']}] {h['doc_id']}  score={h['score']:.3f}")
            snippet = self._extract_snippet(h["file_path"], h["start_line"], h["end_line"])
            for sline in snippet:
                parts.append(f"    {sline}")
        return "\n".join(parts)

    def _build_vector_index(self):
        try:
            import faiss
            import numpy as np
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.warning("faiss or sentence-transformers not installed; vector_search disabled")
            return None

        import logging as _logging
        _logging.getLogger("sentence_transformers").setLevel(_logging.ERROR)
        _logging.getLogger("transformers").setLevel(_logging.ERROR)
        model = SentenceTransformer("all-MiniLM-L6-v2")
        texts: list[str] = []
        ids: list[str] = []
        kinds: list[str] = []

        node_list = list(self.index._nodes.values())[:5000]
        file_paths: list[str] = []
        start_lines: list[int] = []
        end_lines: list[int] = []
        for node in node_list:
            texts.append(f"{node.name} {node.kind}")
            ids.append(node.node_id)
            kinds.append("symbol")
            file_paths.append(node.file_path)
            start_lines.append(node.start_line)
            end_lines.append(node.end_line)

        if not texts:
            return None

        embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        embeddings = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings)
        dim = embeddings.shape[1]
        idx = faiss.IndexFlatIP(dim)
        idx.add(embeddings)

        class _VectorIndex:
            def search(self, query: str, top_k: int = 20) -> list[dict]:
                q_emb = model.encode([query], convert_to_numpy=True).astype(np.float32)
                faiss.normalize_L2(q_emb)
                scores, indices = idx.search(q_emb, top_k)
                results = []
                for score, i in zip(scores[0], indices[0]):
                    if i < 0:
                        continue
                    results.append({
                        "doc_id": ids[i], "kind": kinds[i], "score": float(score),
                        "file_path": file_paths[i],
                        "start_line": start_lines[i],
                        "end_line": end_lines[i],
                    })
                return results

        return _VectorIndex()

    def context_pack(self, symbol: str, token_budget: int = 3000) -> str:
        """
        Pack structured context for a symbol within a token budget.

        Returns: definition snippet + nearest usage sites + related tests
        + imports + enclosing/neighbour symbols.
        """
        parts: list[str] = []
        used = 0

        nodes = self.index.search_symbol(symbol, top_k=1)
        if not nodes:
            return f"Symbol not found: {symbol}"
        node = nodes[0]

        if node.file_path:
            defn = self.open_file(node.file_path, start_line=max(1, node.start_line - 1), n_lines=25)
            parts.append(f"=== Definition: {node.name} ({node.kind}) ===\n{defn}")
            used += len(defn.split())

        if used < token_budget:
            callers = self.index.get_callers(node.node_id, top_k=3)
            for edge in callers:
                src = self.index.get_node(edge.source)
                if src and src.file_path and used < token_budget:
                    snippet = self.open_file(src.file_path, start_line=max(1, src.start_line), n_lines=10)
                    parts.append(f"=== Caller: {src.name} ===\n{snippet}")
                    used += len(snippet.split())

        if used < token_budget:
            file_nodes = self.index.get_file_nodes(node.file_path) if node.file_path else []
            test_nodes = [n for n in file_nodes if "test" in n.name.lower() or "test" in n.file_path.lower()]
            for t in test_nodes[:2]:
                if used < token_budget and t.file_path:
                    snippet = self.open_file(t.file_path, start_line=max(1, t.start_line), n_lines=15)
                    parts.append(f"=== Test: {t.name} ===\n{snippet}")
                    used += len(snippet.split())

        if used < token_budget and node.file_path:
            imports_snippet = self._extract_imports(node.file_path)
            if imports_snippet:
                parts.append(f"=== Imports ({node.file_path}) ===\n{imports_snippet}")

        return "\n\n".join(parts)

    def _extract_imports(self, file_path: str, max_lines: int = 20) -> str:
        full = self.repo_root / file_path
        if not full.exists():
            return ""
        try:
            lines = full.read_text(errors="replace").splitlines()
            import_lines = [l for l in lines[:80] if l.startswith(("import ", "from ", "#include", "require(", "use "))]
            return "\n".join(import_lines[:max_lines])
        except Exception:
            return ""
