import json

import pytest
import requests

from layered_defense import OpenAICompatibleProvider, Pricing, ProviderError, make_provider


class _Resp:
    def __init__(self, status, payload, text=""):
        self.status_code, self._payload, self.text = status, payload, text or json.dumps(payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)


class _Session:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append(json)
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def _ok(text="hi", usage=None):
    return _Resp(200, {"choices": [{"message": {"content": text}}], "usage": usage or {"prompt_tokens": 10, "completion_tokens": 5}, "model": "m"})


def test_auth_failure_raises_provider_error_not_simulation():
    p = OpenAICompatibleProvider("http://x", "k", "m", session=_Session([_Resp(401, {"error": "nope"})]), max_retries=2)
    with pytest.raises(ProviderError, match="authentication"):
        p.chat([{"role": "user", "content": "q"}])


def test_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    s = _Session([requests.ConnectionError("down"), _Resp(500, {"error": "x"}), _ok("ok")])
    p = OpenAICompatibleProvider("http://x", "k", "m", session=s, max_retries=2)
    r = p.chat([{"role": "user", "content": "q"}])
    assert r.text == "ok" and len(s.calls) == 3


def test_gives_up_after_retries(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    s = _Session([requests.ConnectionError("down")] * 3)
    p = OpenAICompatibleProvider("http://x", "k", "m", session=s, max_retries=2)
    with pytest.raises(ProviderError, match="3 attempts"):
        p.chat([{"role": "user", "content": "q"}])


def test_cost_from_api_or_pricing():
    p = OpenAICompatibleProvider("http://x", "k", "m", pricing=Pricing(1.0, 10.0), session=_Session([_ok(), _ok(usage={"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.5})]))
    r1 = p.chat([{"role": "user", "content": "q"}])
    assert r1.cost_source == "pricing" and abs(r1.cost_usd - (10 * 1e-6 + 5 * 10e-6)) < 1e-12
    r2 = p.chat([{"role": "user", "content": "q"}])
    assert r2.cost_source == "api" and r2.cost_usd == 0.5


def test_json_mode_and_model_override_in_payload():
    s = _Session([_ok()])
    OpenAICompatibleProvider("https://openrouter.ai/api/v1", "k", "m", session=s).chat([{"role": "user", "content": "q"}], model="other", json_mode=True)
    payload = s.calls[0]
    assert payload["model"] == "other" and payload["response_format"] == {"type": "json_object"} and payload["usage"] == {"include": True}


def test_make_provider_requires_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ProviderError, match="OPENROUTER_API_KEY"):
        make_provider("openrouter", model="m")
    assert make_provider("mock", model="m").name == "mock"
    with pytest.raises(ProviderError):
        make_provider("nope", model="m")
