"""ExtractorClient: richiesta con JSON schema in formato OpenAI standard."""

import sys
import types

from pipeline.ocr_clients import ExtractorClient


def _fake_openai(monkeypatch, content='{"a": 1}'):
    sent = {}

    class _Completions:
        def create(self, **kw):
            sent.update(kw)
            msg = types.SimpleNamespace(content=content)
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    class OpenAI:
        def __init__(self, **kw):
            self.chat = types.SimpleNamespace(completions=_Completions())

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=OpenAI))
    return sent


def test_extract_uses_response_format_json_schema(monkeypatch):
    sent = _fake_openai(monkeypatch)
    schema = {"type": "object", "properties": {"a": {"type": "integer"}}}
    out = ExtractorClient("http://x/v1", "m").extract("prompt", guided_json_schema=schema)
    assert out == {"a": 1}
    assert sent["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "extraction", "schema": schema},
    }
    assert "guided_json" not in sent.get("extra_body", {})  # rimosso da vLLM
    assert sent["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}


def test_extract_without_schema_has_no_response_format(monkeypatch):
    sent = _fake_openai(monkeypatch)
    ExtractorClient("http://x/v1", "m").extract("prompt")
    assert "response_format" not in sent
