"""LLM feedback providers behind one small interface (feedback.FeedbackProvider).

- ``template``: offline and deterministic (feedback.TemplateProvider). Used by tests and CI.
- ``ollama``: a local model through Ollama's /api/chat.
- ``openai``: any OpenAI-compatible /chat/completions API; the key comes from OPENAI_API_KEY.

No provider is ever required: without configuration the template provider is used, and if an
LLM reply fails the grounding check three times, the template provider takes over.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

import httpx

from .feedback import FeedbackProvider, FeedbackRequest, TemplateProvider


class OllamaProvider:
    name = "ollama"

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "llama3.1:8b",
        timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
    ):
        self.model = model
        # `transport` lets tests plug in httpx.MockTransport instead of a real server.
        self._client = httpx.Client(base_url=host, timeout=timeout, transport=transport)

    def generate(self, request: FeedbackRequest) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": request.messages(),
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
        }
        resp = self._client.post("/api/chat", json=payload)
        resp.raise_for_status()
        return str(resp.json()["message"]["content"])


class OpenAIProvider:
    name = "openai"

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        timeout: float = 60.0,
        transport: httpx.BaseTransport | None = None,
    ):
        self.model = model
        self._client = httpx.Client(
            base_url=base_url,
            timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}"},
            transport=transport,
        )

    def generate(self, request: FeedbackRequest) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": request.messages(),
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        resp = self._client.post("/chat/completions", json=payload)
        resp.raise_for_status()
        return str(resp.json()["choices"][0]["message"]["content"])


def provider_from_env(env: Mapping[str, str] | None = None) -> FeedbackProvider:
    """Pick the provider named in FEEDBACK_PROVIDER (default: template)."""
    env = os.environ if env is None else env
    name = env.get("FEEDBACK_PROVIDER", "template").lower()
    if name == "template":
        return TemplateProvider()
    if name == "ollama":
        return OllamaProvider(
            env.get("OLLAMA_HOST", "http://localhost:11434"),
            env.get("OLLAMA_MODEL", "llama3.1:8b"),
        )
    if name == "openai":
        key = env.get("OPENAI_API_KEY", "")
        if not key:
            raise ValueError("FEEDBACK_PROVIDER=openai needs OPENAI_API_KEY")
        return OpenAIProvider(
            key,
            env.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            env.get("OPENAI_MODEL", "gpt-4o-mini"),
        )
    raise ValueError(f"unknown FEEDBACK_PROVIDER {name!r}: use template, ollama or openai")
