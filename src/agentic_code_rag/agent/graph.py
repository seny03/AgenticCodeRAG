"""
LangGraph state graph for the repository-aware agent.

Graph: plan -> localize -> inspect -> retrieve -> edit -> verify -> (retry | finish)

Each phase runs a tool-call loop: the LLM keeps calling tools until it
emits a phase-transition signal ({"action": "next"}) or a terminal action
({"action": "done"} / {"action": "patch"}).
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Optional

from langgraph.graph import END, StateGraph

from .state import AgentState, ToolCall

logger = logging.getLogger(__name__)


def _strip_code_fences(text: str) -> str:
    """Remove leading/trailing ```python ... ``` fences, preserving body indentation."""
    stripped = text.strip("\n")
    stripped = re.sub(r"^[ \t]*```(?:python|py)?[ \t]*\n", "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"\n[ \t]*```[ \t]*$", "", stripped)
    return stripped

# (signature, one-line description)
_TOOL_DESCRIPTIONS: dict[str, tuple[str, str]] = {
    "repo_tree":   ("repo_tree(path, depth)",
                    "show directory tree — call ONCE in PLAN to orient; do not repeat"),
    "open_file":   ("open_file(path, start_line, n_lines)",
                    "read file lines — use in INSPECT after locating the target; avoid scanning full files in PLAN"),
    "grep_search": ("grep_search(query, glob, top_k)",
                    "text search across files — good for finding imports/usages by literal string"),
    "symbol_search": ("symbol_search(query, kind, top_k)",
                      "semantic symbol lookup — best for finding function/class definitions by name"),
    "goto_definition": ("goto_definition(symbol_or_location)",
                        "jump to definition — use in LOCALIZE to pin exact file and line"),
    "find_references": ("find_references(symbol, top_k)",
                        "all usages of a symbol — use in INSPECT to understand call sites"),
    "find_implementations": ("find_implementations(symbol, top_k)",
                             "concrete implementations of an abstract symbol"),
    "get_callers": ("get_callers(symbol, top_k)",
                    "who calls this function — use in INSPECT to understand context and expected behaviour"),
    "get_callees": ("get_callees(symbol, top_k)",
                    "what this function calls — use in INSPECT to trace dependencies"),
    "graph_neighbors": ("graph_neighbors(node_id, edge_types, hops)",
                        "explore dependency graph around a node"),
    "bm25_search": ("bm25_search(query, top_k)",
                    "keyword search optimised for code retrieval — good in RETRIEVE"),
    "vector_search": ("vector_search(query, top_k)",
                      "semantic similarity search — good for finding analogous implementations"),
    "retrieve_context_for_location": ("retrieve_context_for_location(file_path, line, token_budget)",
                                      "curated context around a specific location — use in RETRIEVE"),
    "retrieve_context_for_task": ("retrieve_context_for_task(task_text, token_budget)",
                                  "curated context for the whole task — best single call in RETRIEVE"),
    "context_pack": ("context_pack(symbol, token_budget)",
                     "full context pack: definition + callers + docs — use in INSPECT or RETRIEVE"),
    "apply_patch": ("apply_patch(unified_diff)",
                    "apply a unified diff to the repo — use in VERIFY to test the patch"),
    "run_lint":    ("run_lint(paths)",
                    "run linter — use in VERIFY"),
    "run_tests":   ("run_tests(selector)",
                    "run tests — use in VERIFY to confirm correctness"),
    "run_command": ("run_command(cmd)",
                    "run an arbitrary shell command"),
    "show_diff":   ("show_diff()",
                    "show current uncommitted diff — use in VERIFY"),
}

_TOOL_SIGNATURES: dict[str, str] = {k: v[0] for k, v in _TOOL_DESCRIPTIONS.items()}


def _build_system_prompt(enabled_tools: list[str] | None) -> str:
    if enabled_tools is not None:
        tool_items = [(t, _TOOL_DESCRIPTIONS[t]) for t in enabled_tools if t in _TOOL_DESCRIPTIONS]
    else:
        tool_items = list(_TOOL_DESCRIPTIONS.items())

    tools_block = "\n".join(
        f"  {sig}\n    → {desc}"
        for _, (sig, desc) in tool_items
    )

    def _t(*names: str) -> str:
        available = [n for n in names if enabled_tools is None or n in enabled_tools]
        return ", ".join(available) if available else "no specialised tools — emit next quickly"

    phase_overview = (
        "Workflow phases (always executed in this order):\n"
        f"  PLAN     — explore repo structure to find relevant areas.\n"
        f"             Tools: {_t('repo_tree', 'grep_search', 'symbol_search')}. Do NOT open full files here.\n"
        f"  LOCALIZE — pinpoint exact files and symbols.\n"
        f"             Tools: {_t('goto_definition', 'find_references', 'symbol_search', 'grep_search')}.\n"
        f"  INSPECT  — read and understand the relevant code.\n"
        f"             Tools: {_t('open_file', 'get_callers', 'get_callees', 'graph_neighbors', 'context_pack')}.\n"
        f"  RETRIEVE — collect all context needed for the edit: callers, similar patterns, type info.\n"
        f"             Tools: {_t('retrieve_context_for_task', 'retrieve_context_for_location', 'bm25_search', 'vector_search', 'context_pack')}.\n"
        "  EDIT     — write the patch or code. Emit the action immediately — do NOT call tools.\n"
        f"  VERIFY   — run tests and lint, then emit done. Tools: {_t('run_tests', 'run_lint', 'show_diff')}.\n"
    )

    return (
        "You are a repository-aware coding agent. You MUST NOT read the entire repository.\n"
        "Access the codebase only through the provided tools.\n\n"
        f"{phase_overview}\n"
        "Available tools:\n"
        f"{tools_block}\n\n"
        "Respond ONLY with JSON (no markdown fences):\n"
        '  {"tool": "<name>", "args": {...}}              <- call a tool\n'
        '  {"action": "next"}                              <- done with this phase, go to next\n'
        '  {"action": "patch", "diff": "<unified diff>"}  <- propose a patch\n'
        '  {"action": "code", "code": "<python code>"}    <- propose a code completion\n'
        '  {"action": "done", "result": "<summary>"}      <- finished, nothing more to do\n'
    )

def _build_phase_instructions(enabled_tools: list[str] | None) -> dict[str, str]:
    """Build phase instructions listing only tools that are actually enabled."""
    def _t(*names: str) -> str:
        available = [n for n in names if enabled_tools is None or n in enabled_tools]
        return ", ".join(available) if available else "(no tools available — emit next)"

    return {
        "plan": (
            f"PLAN: Analyze the task. Use {_t('repo_tree', 'grep_search', 'symbol_search')} to orient. "
            "When you have a picture of what to investigate, emit {\"action\": \"next\"}."
        ),
        "localize": (
            "LOCALIZE: Narrow down to specific files and symbols. "
            f"Use {_t('goto_definition', 'find_references', 'symbol_search', 'grep_search', 'open_file')}. "
            "When you know exactly what to change, emit {\"action\": \"next\"}."
        ),
        "inspect": (
            "INSPECT: Read the relevant code. "
            f"Use {_t('open_file', 'get_callers', 'get_callees', 'graph_neighbors', 'context_pack')}. "
            "Also read a few lines above and below the target function stub — "
            "sibling functions often reveal input-validation patterns (guards, error types, style) "
            "that the implementation must follow. "
            "When you fully understand the code, emit {\"action\": \"next\"}."
        ),
        "retrieve": (
            "RETRIEVE: Gather context needed to make the edit. "
            f"Use {_t('retrieve_context_for_location', 'retrieve_context_for_task', 'bm25_search', 'vector_search', 'context_pack')}. "
            "When context is complete, emit {\"action\": \"next\"}."
        ),
        "edit": (
            "EDIT: Produce a unified diff patch. Emit {\"action\": \"patch\", \"diff\": \"...\"}."
        ),
        "edit_code": (
            "EDIT: Write the complete implementation of the function described in the task. "
            "Use the context you gathered. "
            "You MUST emit {\"action\": \"code\", \"code\": \"...\"} with the full function body "
            "(starting from `def ...`, including the docstring if present). "
            "Do NOT emit {\"action\": \"next\"} in this phase — code is required. "
            "Do NOT emit {\"action\": \"done\"} in this phase — only `code` is accepted."
        ),
        "verify": (
            f"VERIFY: Run tests and lint. Use {_t('run_tests', 'run_lint', 'show_diff')}. "
            "If all pass, emit {\"action\": \"done\", \"result\": \"...\"}. "
            "Otherwise emit {\"action\": \"next\"} to trigger retry."
        ),
    }


def build_agent_graph(
    tool_api: Any,
    llm: Any,
    config: Optional[dict] = None,
    trajectory_dir: Optional[Path] = None,
) -> Any:
    """
    Build and compile the LangGraph StateGraph.

    Parameters
    ----------
    tool_api : DraCoToolAPI
        The tool API instance.
    llm : langchain BaseChatModel
        The LLM to use for reasoning.
    config : dict, optional
        Agent configuration overrides.
    trajectory_dir : Path, optional
        Directory to save trajectory JSON files.
    """
    cfg = config or {}
    max_retries = cfg.get("max_retries", 3)
    max_tool_calls_per_phase = cfg.get("max_tool_calls_per_phase", 15)
    max_total_tool_calls = cfg.get("max_total_tool_calls", 80)
    enabled_tools: list[str] | None = cfg.get("enabled_tools", None)
    api_max_retries = cfg.get("api_max_retries", 7)
    api_retry_init_delay = cfg.get("api_retry_init_delay", 1.0)

    system_prompt = _build_system_prompt(enabled_tools)
    phase_instructions = _build_phase_instructions(enabled_tools)

    full_dispatch = _build_dispatch(tool_api)
    if enabled_tools is not None:
        tool_dispatch = {k: v for k, v in full_dispatch.items() if k in enabled_tools}
    else:
        tool_dispatch = full_dispatch

    def _call_llm(state: AgentState, instruction: str) -> dict:
        state.system_prompt = system_prompt
        messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Task: {state.task_text}\n\n{instruction}"},
        ]
        for tc in state.trajectory[-30:]:
            messages.append({"role": "assistant", "content": json.dumps({
                "tool": tc.tool_name, "args": tc.arguments,
            })})
            messages.append({"role": "user", "content": f"[tool result: {tc.tool_name}]\n{tc.result[:3000]}"})

        if state.retrieved_context:
            messages.append({
                "role": "user",
                "content": f"Accumulated context:\n{state.retrieved_context[-12000:]}",
            })

        delay = api_retry_init_delay
        for attempt in range(api_max_retries + 1):
            try:
                response = llm.invoke(messages)
                content = response.content if hasattr(response, "content") else str(response)
                state.llm_turns.append({
                    "phase": state.phase,
                    "instruction": instruction,
                    "raw": content,
                    "retrieved_context": state.retrieved_context,
                })
                return _parse_json(content)
            except Exception as exc:
                if attempt >= api_max_retries:
                    logger.error("LLM call failed after %d retries: %s", api_max_retries, exc)
                    return {"action": "next"}
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s — retrying in %.1fs",
                    attempt + 1, api_max_retries, exc, delay,
                )
                time.sleep(delay)
                delay *= 2

    def _execute_tool(state: AgentState, tool_name: str, args: dict, phase: str = "") -> str:
        handler = tool_dispatch.get(tool_name)
        if handler is None:
            return f"Unknown tool: {tool_name}"
        t0 = time.time()
        try:
            result = str(handler(args))
        except Exception as exc:
            result = f"Tool error ({tool_name}): {exc}"
        tc = ToolCall(
            tool_name=tool_name,
            arguments=args,
            result=result[:2000],
            timestamp=t0,
            phase=phase,
            duration_s=round(time.time() - t0, 2),
        )
        state.trajectory.append(tc)
        state.total_tool_calls += 1
        return result

    def _run_phase(state: AgentState, phase_name: str) -> dict:
        instruction = phase_instructions[phase_name]
        if phase_name == "edit" and state.output_format == "code":
            instruction = phase_instructions["edit_code"]
        phase_calls = 0
        # Seed context_parts from previous phases so context accumulates.
        context_parts: list[str] = (
            [state.retrieved_context] if state.retrieved_context else []
        )
        last_call_sig: Optional[tuple] = None
        repeat_count = 0

        while True:
            if max_total_tool_calls and state.total_tool_calls >= max_total_tool_calls:
                state.finished = True
                state.error = "Max total tool calls exceeded"
                return _state_dict(state, "finish")

            if max_tool_calls_per_phase and phase_calls >= max_tool_calls_per_phase:
                break

            resp = _call_llm(state, instruction)
            action = resp.get("action", "")

            if action == "next":
                break

            if action == "done":
                state.finished = True
                _save_trajectory(state, trajectory_dir)
                return _state_dict(state, "finish")

            if action == "patch":
                if phase_name != "edit":
                    break  # premature patch — advance to next phase
                diff = resp.get("diff", "")
                result = _execute_tool(state, "apply_patch", {"unified_diff": diff}, phase=phase_name)
                state.proposed_patch = diff
                context_parts.append(f"[apply_patch] {result[:500]}")
                phase_calls += 1
                state.retrieved_context = "\n\n".join(context_parts)
                _save_trajectory(state, trajectory_dir)
                return _state_dict(state, "verify")

            if action == "code":
                if phase_name != "edit":
                    break  # premature code — advance to next phase
                code = resp.get("code", "")
                state.proposed_patch = code
                state.finished = True
                state.retrieved_context = "\n\n".join(context_parts)
                _save_trajectory(state, trajectory_dir)
                return _state_dict(state, "finish")

            tool_name = resp.get("tool", "")
            args = resp.get("args", {})
            if not tool_name:
                logger.warning("LLM returned unexpected response: %s", resp)
                break

            try:
                call_sig = (tool_name, json.dumps(args, sort_keys=True, default=str))
            except Exception:
                call_sig = (tool_name, repr(args))
            if call_sig == last_call_sig:
                repeat_count += 1
            else:
                repeat_count = 0
                last_call_sig = call_sig
            if repeat_count >= 2:
                logger.info(
                    "Phase %s: same %s call repeated 3x — advancing to next phase",
                    phase_name, tool_name,
                )
                context_parts.append(
                    f"[note] You called {tool_name} with the same args 3 times — "
                    f"the result will not change. Moving to the next phase."
                )
                break

            result = _execute_tool(state, tool_name, args, phase=phase_name)
            context_parts.append(f"[{tool_name}]\n{result[:1500]}")
            phase_calls += 1

            if tool_name in ("symbol_search", "grep_search", "bm25_search", "vector_search"):
                for line in result.splitlines():
                    if ":" in line and "/" in line:
                        path = line.split(":")[0].strip()
                        if path and path not in state.candidate_files:
                            state.candidate_files.append(path)

        state.retrieved_context = "\n\n".join(context_parts)

        # Fallback: in code-completion mode, edit phase must produce code.
        # If the model emitted "next" or hit the call limit without emitting code,
        # ask once more with a hard-prompt for code only.
        if (
            phase_name == "edit"
            and state.output_format == "code"
            and not state.proposed_patch
        ):
            fallback_code = _fallback_code_request(state)
            if fallback_code:
                state.proposed_patch = fallback_code
                state.finished = True
                _save_trajectory(state, trajectory_dir)
                return _state_dict(state, "finish")

        next_phase = _next_phase(phase_name)
        _save_trajectory(state, trajectory_dir)
        return _state_dict(state, next_phase)

    def _fallback_code_request(state: AgentState) -> str:
        """Last-resort direct LLM call asking only for the function body."""
        traj_summary = "\n\n".join(
            f"[{tc.tool_name}({json.dumps(tc.arguments)[:120]})]\n{tc.result[:1500]}"
            for tc in state.trajectory[-15:]
        )
        fallback_messages = [
            {
                "role": "system",
                "content": (
                    "You are a code completion assistant. "
                    "Return ONLY the complete function implementation. "
                    "Start with the `def` line (function signature), include the docstring "
                    "if present, then the full body. "
                    "Do not include imports, other functions, markdown fences, or any explanation."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Task / function to implement:\n{state.task_text}\n\n"
                    f"Context gathered so far (tool results):\n{traj_summary}\n\n"
                    "Now write the complete function body. Return only the code."
                ),
            },
        ]
        delay = api_retry_init_delay
        for attempt in range(api_max_retries + 1):
            try:
                response = llm.invoke(fallback_messages)
                content = response.content if hasattr(response, "content") else str(response)
                return _strip_code_fences(content).strip()
            except Exception as exc:
                if attempt >= api_max_retries:
                    logger.error("Fallback LLM call failed: %s", exc)
                    return ""
                time.sleep(delay)
                delay *= 2
        return ""

    def plan_node(state: AgentState) -> dict:
        return _run_phase(state, "plan")

    def localize_node(state: AgentState) -> dict:
        return _run_phase(state, "localize")

    def inspect_node(state: AgentState) -> dict:
        return _run_phase(state, "inspect")

    def retrieve_node(state: AgentState) -> dict:
        return _run_phase(state, "retrieve")

    def edit_node(state: AgentState) -> dict:
        return _run_phase(state, "edit")

    def verify_node(state: AgentState) -> dict:
        result = _run_phase(state, "verify")
        state.verification_result = state.retrieved_context
        return result

    def retry_node(state: AgentState) -> dict:
        state.retry_count += 1
        if state.retry_count >= max_retries:
            state.finished = True
            state.error = f"Max retries ({max_retries}) exceeded"
            _save_trajectory(state, trajectory_dir)
            return _state_dict(state, "finish")
        return _state_dict(state, "edit")

    def finish_node(state: AgentState) -> dict:
        state.finished = True
        _save_trajectory(state, trajectory_dir)
        return {"finished": True}

    def route_after_edit(state: AgentState) -> str:
        return "finish" if state.finished else "verify"

    def should_retry(state: AgentState) -> str:
        if state.finished:
            return "finish"
        vr = state.verification_result.lower()
        if any(w in vr for w in ("failed", "error", "fail", "assert")):
            return "retry"
        return "finish"

    graph = StateGraph(AgentState)
    graph.add_node("plan", plan_node)
    graph.add_node("localize", localize_node)
    graph.add_node("inspect", inspect_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("edit", edit_node)
    graph.add_node("verify", verify_node)
    graph.add_node("retry", retry_node)
    graph.add_node("finish", finish_node)

    graph.set_entry_point("plan")
    graph.add_edge("plan", "localize")
    graph.add_edge("localize", "inspect")
    graph.add_edge("inspect", "retrieve")
    graph.add_edge("retrieve", "edit")
    graph.add_conditional_edges("edit", route_after_edit, {"verify": "verify", "finish": "finish"})
    graph.add_conditional_edges("verify", should_retry, {"retry": "retry", "finish": "finish"})
    graph.add_edge("retry", "edit")
    graph.add_edge("finish", END)

    return graph.compile()


def _next_phase(current: str) -> str:
    order = ["plan", "localize", "inspect", "retrieve", "edit", "verify", "finish"]
    idx = order.index(current) if current in order else -1
    return order[idx + 1] if idx + 1 < len(order) else "finish"


def _state_dict(state: AgentState, phase: str) -> dict:
    return {
        "phase": phase,
        "finished": state.finished,
        "error": state.error,
        "candidate_files": state.candidate_files,
        "candidate_symbols": state.candidate_symbols,
        "retrieved_context": state.retrieved_context,
        "proposed_patch": state.proposed_patch,
        "verification_result": state.verification_result,
        "trajectory": state.trajectory,
        "total_tool_calls": state.total_tool_calls,
        "retry_count": state.retry_count,
        "output_format": state.output_format,
        "system_prompt": state.system_prompt,
        "llm_turns": state.llm_turns,
    }


def _parse_json(content: str) -> dict:
    content = content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(content[start:end])
            except json.JSONDecodeError:
                pass
    return {"action": "next"}


def _save_trajectory(state: AgentState, trajectory_dir: Optional[Path]) -> None:
    if not trajectory_dir:
        return
    import datetime
    trajectory_dir = Path(trajectory_dir)
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    stem = trajectory_dir.name

    tc_list = [
        {
            "phase": tc.phase,
            "tool_name": tc.tool_name,
            "arguments": tc.arguments,
            "result": tc.result,
            "timestamp": tc.timestamp,
            "duration_s": tc.duration_s,
        }
        for tc in state.trajectory
    ]
    payload = {
        "task_text": state.task_text,
        "repo_root": state.repo_root,
        "snapshot_id": state.snapshot_id,
        "finished": state.finished,
        "error": state.error,
        "candidate_files": state.candidate_files,
        "proposed_patch": state.proposed_patch,
        "verification_result": state.verification_result,
        "total_tool_calls": state.total_tool_calls,
        "retry_count": state.retry_count,
        "trajectory": tc_list,
        "llm_turns": state.llm_turns or [],
    }
    (trajectory_dir / f"{stem}.json").write_text(json.dumps(payload, indent=2, default=str))
    (trajectory_dir / f"{stem}.md").write_text(_render_markdown(state, tc_list))
    logger.debug("Trajectory saved to %s", stem)


def _render_markdown(state: AgentState, tc_list: list[dict]) -> str:
    import datetime
    task_preview = state.snapshot_id or state.task_text[:80].replace("\n", " ").strip()
    status = "✅ finished" if state.finished and not state.error else (
        f"❌ {state.error}" if state.error else "⏳ in progress"
    )
    t_start = tc_list[0]["timestamp"] if tc_list else time.time()
    t_end = tc_list[-1]["timestamp"] + tc_list[-1]["duration_s"] if tc_list else t_start
    elapsed = round(t_end - t_start, 1)
    started_at = datetime.datetime.fromtimestamp(t_start).strftime("%Y-%m-%d %H:%M:%S")

    lines: list[str] = [
        f"# Trajectory: {task_preview}",
        f"",
        f"**Status:** {status} | **Tool calls:** {state.total_tool_calls}"
        f" | **Retries:** {state.retry_count} | **Elapsed:** {elapsed}s | **Started:** {started_at}",
        f"",
        f"---",
        f"## System Prompt",
        f"```",
        state.system_prompt.strip() if state.system_prompt else "(not recorded)",
        f"```",
        f"",
        f"## Task",
        f"```",
        state.task_text.strip(),
        f"```",
    ]

    if state.candidate_files:
        lines += ["", "**Candidate files:** " + ", ".join(f"`{f}`" for f in state.candidate_files[:10])]

    # Collect phases that actually ran, render in predefined order
    _phase_order = ["plan", "localize", "inspect", "retrieve", "edit", "verify", "finish"]
    seen_phase_set: set[str] = set()
    for tc in tc_list:
        seen_phase_set.add(tc["phase"])
    for lt in (state.llm_turns or []):
        seen_phase_set.add(lt["phase"])
    seen_phases = [p for p in _phase_order if p in seen_phase_set]
    for p in seen_phase_set:
        if p not in seen_phases:
            seen_phases.append(p)

    llm_turns = state.llm_turns or []

    for phase in seen_phases:
        phase_tcs = [tc for tc in tc_list if tc["phase"] == phase]
        phase_llm = [lt for lt in llm_turns if lt["phase"] == phase]

        lines += ["", "---", f"## Phase: {phase.upper()}"]

        if phase_llm:
            lines += ["", f"**Instruction:** {phase_llm[0].get('instruction', '')}"]

        # Interleave: LLM turn -> tool call -> ... -> last LLM turn (action)
        tc_idx = 0
        for i, lt in enumerate(phase_llm):
            # Show accumulated context before the first turn where it's non-empty
            if i == 0:
                ctx = lt.get("retrieved_context", "")
                if ctx:
                    lines += ["", "**Accumulated context passed to model:**", "```", ctx, "```"]

            raw = lt.get("raw", "")
            lines += ["", f"### LLM turn {i + 1}", "```json", raw, "```"]

            if tc_idx < len(phase_tcs):
                tc = phase_tcs[tc_idx]
                args_str = ", ".join(f"{k}={json.dumps(v)}" for k, v in tc["arguments"].items())
                duration = f"  _{tc['duration_s']}s_" if tc["duration_s"] else ""
                lines += ["", f"→ **`{tc['tool_name']}({args_str})`**{duration}"]
                result = tc["result"].strip()
                lines += ["```", result, "```"]
                tc_idx += 1

        # Orphan tool calls (no matching LLM turn recorded)
        while tc_idx < len(phase_tcs):
            tc = phase_tcs[tc_idx]
            args_str = ", ".join(f"{k}={json.dumps(v)}" for k, v in tc["arguments"].items())
            result = tc["result"].strip()
            lines += ["", f"→ **`{tc['tool_name']}({args_str})`**", "```", result, "```"]
            tc_idx += 1

    if state.proposed_patch:
        if getattr(state, "output_format", "patch") == "code":
            lines += ["", "---", "## Generated Code", "```python", state.proposed_patch, "```"]
        else:
            lines += ["", "---", "## Proposed Patch", "```diff", state.proposed_patch, "```"]

    if state.verification_result:
        lines += ["", "---", "## Verification", "```", state.verification_result, "```"]

    return "\n".join(lines) + "\n"


def _build_dispatch(tool_api: Any) -> dict:
    return {
        "repo_tree": lambda a: tool_api.repo_tree(a.get("path", "."), a.get("depth", 3)),
        "open_file": lambda a: tool_api.open_file(a["path"], a.get("start_line", 1), a.get("n_lines", 100)),
        "grep_search": lambda a: tool_api.grep_search(a["query"], a.get("glob", "**/*"), a.get("top_k", 20)),
        "symbol_search": lambda a: tool_api.symbol_search(a["query"], a.get("kind"), a.get("top_k", 20)),
        "goto_definition": lambda a: tool_api.goto_definition(a["symbol_or_location"]),
        "find_references": lambda a: tool_api.find_references(a["symbol"], a.get("top_k", 20)),
        "find_implementations": lambda a: tool_api.find_implementations(a["symbol"], a.get("top_k", 20)),
        "get_callers": lambda a: tool_api.get_callers(a["symbol"], a.get("top_k", 20)),
        "get_callees": lambda a: tool_api.get_callees(a["symbol"], a.get("top_k", 20)),
        "graph_neighbors": lambda a: tool_api.graph_neighbors(a["node_id"], a.get("edge_types"), a.get("hops", 1)),
        "bm25_search": lambda a: tool_api.bm25_search(a["query"], a.get("top_k", 20)),
        "vector_search": lambda a: tool_api.vector_search(a["query"], a.get("top_k", 20)),
        "retrieve_context_for_location": lambda a: tool_api.retrieve_context_for_location(a["file_path"], a["line"], a.get("token_budget", 4000)),
        "retrieve_context_for_task": lambda a: tool_api.retrieve_context_for_task(a["task_text"], a.get("token_budget", 4000)),
        "context_pack": lambda a: tool_api.context_pack(a["symbol"], a.get("token_budget", 3000)),
        "apply_patch": lambda a: tool_api.apply_patch(a["unified_diff"]),
        "run_lint": lambda a: tool_api.run_lint(a.get("paths")),
        "run_tests": lambda a: tool_api.run_tests(a.get("selector", "")),
        "run_command": lambda a: tool_api.run_command(a["cmd"]),
        "show_diff": lambda a: tool_api.show_diff(),
    }
