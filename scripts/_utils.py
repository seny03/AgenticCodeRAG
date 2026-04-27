"""Shared utilities for runner scripts."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Optional


COMPLETION_SYSTEM_PROMPT = (
    "You are a code completion assistant. "
    "Return ONLY the complete function implementation. "
    "Always start your response with the `def` line (function signature), "
    "even if the signature already appears in the context — repeat it. "
    "Include the docstring if present, then the full body. "
    "Do not include imports, other functions, markdown fences, or any explanation."
)


def _strip_markdown_fences(text: str) -> str:
    """Remove leading/trailing ```python ... ``` fences if present.

    Only consumes horizontal whitespace + a single newline around the fences,
    so the body's own leading indentation is preserved.
    """
    stripped = text.strip("\n")
    stripped = re.sub(r"^[ \t]*```(?:python|py)?[ \t]*\n", "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"\n[ \t]*```[ \t]*$", "", stripped)
    return stripped


def _extract_def_signature(prompt: str) -> Optional[str]:
    """Find the last `def ...` line (and its docstring, if any) in the prompt."""
    lines = prompt.splitlines()
    def_idx = None
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].lstrip().startswith("def "):
            def_idx = i
            break
    if def_idx is None:
        return None
    return "\n".join(lines[def_idx:])


def postprocess_completion(prompt: str, output: str) -> str:
    """
    Clean up completion model output:
      1. Strip surrounding ```python ... ``` markdown fences.
      2. If output doesn't start with `def`, prepend the `def` line + docstring
         from the prompt so process_result.py can parse it.
    """
    cleaned = _strip_markdown_fences(output)
    if cleaned.lstrip().startswith("def "):
        return cleaned
    signature = _extract_def_signature(prompt)
    if signature is None:
        return cleaned
    return signature + "\n" + cleaned


def load_setup_config(path: str | Path | None) -> dict[str, Any]:
    """
    Load a setup YAML config file and return its contents as a dict.

    Returns an empty dict if path is None.
    """
    if path is None:
        return {}
    import yaml

    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return data


def agent_cfg_from_setup(setup: dict[str, Any]) -> dict[str, Any]:
    """
    Extract the agent configuration sub-dict from a setup YAML dict.

    The returned dict is suitable for passing directly to build_agent_graph().
    """
    cfg: dict[str, Any] = {}
    agent_block = setup.get("agent", {})
    for key in ("max_retries", "max_tool_calls_per_phase", "max_total_tool_calls"):
        if key in agent_block:
            cfg[key] = agent_block[key]
    enabled = setup.get("enabled_tools")
    if enabled is not None:
        cfg["enabled_tools"] = enabled
    return cfg


def build_generate_fn(
    llm: Any,
    setup: dict[str, Any],
    tool_api: Optional[Any] = None,
    task_hint: str = "",
    context_capture: Optional[list] = None,
    raw_capture: Optional[list] = None,
) -> Optional[Callable[[str], str]]:
    """
    Return a generate_fn appropriate for the setup mode, or None if the setup
    should use the full LangGraph agent.

    mode: zero_shot
        Direct call to the LLM with the prompt as-is.  No retrieval.

    mode: static_rag
        Retrieve context programmatically (BM25 + DraCo context_pack) and
        prepend it to the prompt, then call the LLM once.

    mode: anything else (or missing)
        Return None — the caller should use the full agent.

    Parameters
    ----------
    llm:
        LangChain BaseChatModel instance.
    setup:
        Loaded setup YAML dict (may be empty).
    tool_api:
        DraCoToolAPI instance (needed for static_rag retrieval).
    task_hint:
        Short text used as the retrieval query for static_rag.
    """
    mode = setup.get("mode", "agent")

    if mode == "zero_shot":
        def _zero_shot(prompt: str) -> str:
            response = llm.invoke([
                {"role": "system", "content": COMPLETION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ])
            raw = response.content if hasattr(response, "content") else str(response)
            if raw_capture is not None:
                raw_capture.clear()
                raw_capture.append(raw)
            return postprocess_completion(prompt, raw)
        return _zero_shot

    if mode == "static_rag":
        retrieval_cfg = setup.get("retrieval", {})
        bm25_top_k: int = retrieval_cfg.get("bm25_top_k", 5)
        token_budget: int = retrieval_cfg.get("token_budget", 4000)
        use_draco: bool = retrieval_cfg.get("draco_context", True)

        def _static_rag(prompt: str) -> str:
            context_parts: list[str] = []

            if tool_api is not None:
                # For code-completion prompts the first 300 chars are imports;
                # the target signature + docstring at the end is a far better
                # retrieval query.
                query = task_hint or _extract_def_signature(prompt) or prompt[:300]
                # Cap to avoid blowing past embedding-model token limits.
                query = query[:1500]
                if bm25_top_k > 0:
                    bm25_result = tool_api.bm25_search(query, top_k=bm25_top_k)
                    if bm25_result.strip():
                        context_parts.append(f"[BM25 results]\n{bm25_result}")
                if use_draco:
                    draco_result = tool_api.retrieve_context_for_task(query, token_budget=token_budget)
                    if draco_result.strip():
                        context_parts.append(f"[Repository context]\n{draco_result}")

            if context_parts:
                context_block = "\n\n".join(context_parts)
                augmented = f"{context_block}\n\n---\n\n{prompt}"
            else:
                augmented = prompt

            if context_capture is not None:
                context_capture.clear()
                context_capture.append(augmented)

            response = llm.invoke([
                {"role": "system", "content": COMPLETION_SYSTEM_PROMPT},
                {"role": "user", "content": augmented},
            ])
            raw = response.content if hasattr(response, "content") else str(response)
            if raw_capture is not None:
                raw_capture.clear()
                raw_capture.append(raw)
            return postprocess_completion(prompt, raw)

        return _static_rag

    return None
