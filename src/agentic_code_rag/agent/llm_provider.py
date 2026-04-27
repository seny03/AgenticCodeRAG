"""
LLM provider factory.

Supports three provider modes, all reading configuration from environment
variables

Anthropic::

    ANTHROPIC_API_KEY=<token>
    ANTHROPIC_BASE_URL=<base_url> # optional
    ANTHROPIC_MODEL=<model>

OpenAI::

    OPENAI_API_KEY=<token>

OpenAI-compatible server (including local e.g. ollama, vLLM)::

    OPENAI_COMPATIBLE_BASE_URL=<base_url>
    OPENAI_COMPATIBLE_API_KEY=<token> # optional
    OPENAI_COMPATIBLE_MODEL=<model>

TLS / proxy (applied to every provider)::

    HTTPS_PROXY=<proxy url> # optional
    SSL_CERT_FILE=/path/to/ca-bundle.pem # optional

Usage::

    from agentic_code_rag.agent.llm_provider import create_llm

    llm = create_llm("anthropic", model="claude-sonnet-4-6")
    llm = create_llm("openai", model="gpt-4o")
    llm = create_llm("openai_compatible", model="qwen2.5-coder:0.5b",
                      base_url="http://localhost:11434/v1")
"""

from __future__ import annotations

import os
from typing import Any, Optional


def _make_http_client() -> Any:
    """
    Build an httpx.Client that honours HTTPS_PROXY and SSL_CERT_FILE from env.

    Returns None if httpx is not installed.
    """
    try:
        import httpx
    except ImportError:
        return None

    proxy_url: Optional[str] = (
        os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy") or None
    )
    cert_file: Optional[str] = os.environ.get("SSL_CERT_FILE") or None

    kwargs: dict[str, Any] = {"timeout": 60}
    if proxy_url:
        kwargs["proxy"] = proxy_url
    if cert_file:
        kwargs["verify"] = cert_file

    return httpx.Client(**kwargs)


def create_llm(
    provider: str,
    model: str,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 16384,
    **kwargs: Any,
) -> Any:
    """
    Create a LangChain-compatible chat model.

    Parameters
    ----------
    provider :
        One of "openai", "anthropic", "openai_compatible", "local".
    model :
        Model name/identifier.
    base_url :
        Base URL override.  When omitted, read from env:
        - anthropic -> ANTHROPIC_BASE_URL (optional)
        - openai_compatible / local -> OPENAI_COMPATIBLE_BASE_URL
    api_key :
        API key override.  When omitted, read from env:
        - anthropic -> ANTHROPIC_API_KEY
        - openai -> OPENAI_API_KEY
        - openai_compatible / local -> OPENAI_COMPATIBLE_API_KEY
    temperature :
        Sampling temperature.
    max_tokens :
        Maximum tokens in the response.
    """
    provider = provider.lower().strip()

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        key: str = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        effective_base = base_url or os.environ.get("ANTHROPIC_BASE_URL", "")
        extra: dict[str, Any] = {}
        if effective_base:
            extra["base_url"] = effective_base
        http_client = _make_http_client()
        if http_client is not None:
            extra["http_client"] = http_client
        return ChatAnthropic(
            model=model,
            api_key=key,
            temperature=temperature,
            max_tokens=max_tokens,
            **{**extra, **kwargs},
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        extra = {}
        http_client = _make_http_client()
        if http_client is not None:
            extra["http_client"] = http_client
        return ChatOpenAI(
            model=model,
            api_key=key,
            temperature=temperature,
            max_tokens=max_tokens,
            **{**extra, **kwargs},
        )

    if provider in ("openai_compatible", "local"):
        from langchain_openai import ChatOpenAI

        url: str = (
            base_url
            or os.environ.get("OPENAI_COMPATIBLE_BASE_URL", "http://localhost:8000/v1")
        )
        key = api_key or os.environ.get("OPENAI_COMPATIBLE_API_KEY", "none")
        extra = {}
        http_client = _make_http_client()
        if http_client is not None:
            extra["http_client"] = http_client
        return ChatOpenAI(
            model=model,
            base_url=url,
            api_key=key,
            temperature=temperature,
            max_tokens=max_tokens,
            **{**extra, **kwargs},
        )

    raise ValueError(
        f"Unknown provider '{provider}'. "
        "Supported: openai, anthropic, openai_compatible, local"
    )
