import json

import httpx
import pytest

from essay_scorer import CRITERIA
from essay_scorer.feedback import FeedbackRequest, TemplateProvider
from essay_scorer.providers import OllamaProvider, OpenAIProvider, provider_from_env

BANDS = {**{c: 6.0 for c in CRITERIA}, "overall": 6.0}
REQUEST = FeedbackRequest(
    "An essay with <b>tags</b> inside it. It is short.", "A prompt here.", BANDS
)


def test_ollama_provider_sends_escaped_messages_and_reads_the_reply():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"], seen["body"] = request.url.path, json.loads(request.content)
        return httpx.Response(200, json={"message": {"content": '{"ok": true}'}})

    provider = OllamaProvider(model="llama3.1:8b", transport=httpx.MockTransport(handler))
    assert provider.generate(REQUEST) == '{"ok": true}'
    assert seen["path"] == "/api/chat" and seen["body"]["format"] == "json"
    assert "&lt;b&gt;tags" in seen["body"]["messages"][1]["content"]


def test_openai_provider_sends_key_and_json_mode():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"], seen["body"] = request.headers["authorization"], json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    provider = OpenAIProvider("sk-test", transport=httpx.MockTransport(handler))
    assert provider.generate(REQUEST) == "{}"
    assert seen["auth"] == "Bearer sk-test"
    assert seen["body"]["response_format"] == {"type": "json_object"}
    assert seen["body"]["temperature"] == 0


def test_provider_from_env_never_requires_a_key():
    assert isinstance(provider_from_env({}), TemplateProvider)
    assert isinstance(provider_from_env({"FEEDBACK_PROVIDER": "ollama"}), OllamaProvider)
    keyed = provider_from_env({"FEEDBACK_PROVIDER": "openai", "OPENAI_API_KEY": "k"})
    assert isinstance(keyed, OpenAIProvider)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        provider_from_env({"FEEDBACK_PROVIDER": "openai"})
    with pytest.raises(ValueError):
        provider_from_env({"FEEDBACK_PROVIDER": "gpt"})
