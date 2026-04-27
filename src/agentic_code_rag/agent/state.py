"""
Agent state definition for the LangGraph state machine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional


@dataclass
class ToolCall:
    tool_name: str
    arguments: dict[str, Any]
    result: str = ""
    timestamp: float = 0.0
    phase: str = ""
    duration_s: float = 0.0


@dataclass
class AgentState:
    task_text: str = ""
    repo_root: str = ""
    snapshot_id: str = ""

    candidate_files: list[str] = field(default_factory=list)
    candidate_symbols: list[str] = field(default_factory=list)
    retrieved_context: str = ""
    proposed_patch: str = ""
    verification_result: str = ""

    trajectory: list[ToolCall] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)

    phase: str = "plan"
    retry_count: int = 0
    max_retries: int = 3
    finished: bool = False
    error: str = ""

    output_format: Literal["patch", "code"] = "patch"

    system_prompt: str = ""
    llm_turns: list[dict] = field(default_factory=list)

    token_budget: int = 8000
    total_tool_calls: int = 0
    total_context_tokens: int = 0
