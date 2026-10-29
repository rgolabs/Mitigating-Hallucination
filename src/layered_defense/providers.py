"""Chat-completion providers.

Only one network provider is needed: anything that speaks the OpenAI
chat-completions protocol (OpenRouter, OpenAI, Groq, Ollama, Google Gemini's
OpenAI-compatible endpoint, vLLM, ...).  :class:`MockProvider` is a
deterministic offline stand-in for tests and dry runs.  A failed call raises
:class:`ProviderError`; it is never replaced by simulated output.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import requests

from .corpus import ADVERSARIAL_DOCS


class ProviderError(RuntimeError):
    """Raised when a provider cannot return a usable completion."""


@dataclass(frozen=True)
class Pricing:
    """USD per one million tokens.  Used when the API does not report cost."""

    input_per_m: float = 0.0
    output_per_m: float = 0.0

    def cost(self, prompt_tokens: float, completion_tokens: float) -> float:
        return prompt_tokens * self.input_per_m / 1e6 + completion_tokens * self.output_per_m / 1e6


@dataclass
class ChatResponse:
    text: str
    latency_s: float
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    cost_source: str = "unknown"  # "api" | "pricing" | "mock"
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


class ChatProvider(Protocol):
    name: str

    def chat(
        self,
        messages: Sequence[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> ChatResponse: ...


KNOWN_BASE_URLS = {
    "openrouter": "https://openrouter.ai/api/v1",
    "openai": "https://api.openai.com/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
    "groq": "https://api.groq.com/openai/v1",
    "ollama": "http://localhost:11434/v1",
}
API_KEY_ENV = {
    "openrouter": "OPENROUTER_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
    "ollama": "OLLAMA_API_KEY",
}


class OpenAICompatibleProvider:
    """Client for any OpenAI-compatible ``/chat/completions`` endpoint."""

    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        model: str,
        *,
        name: str = "openai-compatible",
        pricing: Pricing | None = None,
        timeout: float = 60.0,
        max_retries: int = 2,
        retry_backoff_s: float = 2.0,
        extra_headers: dict[str, str] | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.name = name
        self.pricing = pricing or Pricing()
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff_s = retry_backoff_s
        self.extra_headers = extra_headers or {}
        self.session = session or requests.Session()

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json", **self.extra_headers}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def chat(
        self,
        messages: Sequence[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": list(messages),
            "temperature": temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if "openrouter.ai" in self.base_url:
            payload["usage"] = {"include": True}  # ask OpenRouter to report cost

        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            start = time.perf_counter()
            try:
                res = self.session.post(
                    f"{self.base_url}/chat/completions",
                    headers=self._headers(),
                    json=payload,
                    timeout=self.timeout,
                )
                latency = time.perf_counter() - start
                if res.status_code in (401, 403):
                    raise ProviderError(f"{self.name}: authentication failed (HTTP {res.status_code}). Check your API key.")
                if res.status_code == 429 or res.status_code >= 500:
                    raise requests.HTTPError(f"HTTP {res.status_code}: {res.text[:200]}", response=res)
                res.raise_for_status()
                data = res.json()
                return self._parse(data, payload["model"], latency)
            except ProviderError:
                raise
            except (requests.RequestException, ValueError, KeyError, IndexError) as err:
                last_err = err
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff_s * (2**attempt))
        raise ProviderError(f"{self.name}: request failed after {self.max_retries + 1} attempts: {last_err}") from last_err

    def stream_chat(
        self,
        messages: Sequence[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        """Yield content deltas via server-sent events; returns the final ChatResponse."""
        payload: dict[str, Any] = {"model": model or self.model, "messages": list(messages), "temperature": temperature, "stream": True, "stream_options": {"include_usage": True}}
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if max_tokens:
            payload["max_tokens"] = max_tokens
        start = time.perf_counter()
        res = self.session.post(f"{self.base_url}/chat/completions", headers=self._headers(), json=payload, timeout=self.timeout, stream=True)
        if res.status_code in (401, 403):
            raise ProviderError(f"{self.name}: authentication failed (HTTP {res.status_code}). Check your API key.")
        if res.status_code >= 400:
            raise ProviderError(f"{self.name}: HTTP {res.status_code}: {res.text[:200]}")
        parts: list[str] = []
        usage: dict[str, Any] = {}
        model_id = payload["model"]
        for raw in res.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data:"):
                continue
            body = raw[5:].strip()
            if body == "[DONE]":
                break
            try:
                chunk = json.loads(body)
            except json.JSONDecodeError:
                continue
            model_id = chunk.get("model") or model_id
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices") or []:
                delta = (choice.get("delta") or {}).get("content")
                if delta:
                    parts.append(delta)
                    yield delta
        pt, ct = int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)
        cost, source = (float(usage["cost"]), "api") if usage.get("cost") is not None else (self.pricing.cost(pt, ct), "pricing")
        return ChatResponse("".join(parts).strip(), time.perf_counter() - start, str(model_id), pt, ct, cost, source)

    def _parse(self, data: dict[str, Any], model: str, latency: float) -> ChatResponse:
        if "error" in data and not data.get("choices"):
            raise ProviderError(f"{self.name}: API error: {data['error']}")
        choice = data["choices"][0]
        text = (choice.get("message") or {}).get("content") or ""
        usage = data.get("usage") or {}
        pt = int(usage.get("prompt_tokens") or 0)
        ct = int(usage.get("completion_tokens") or 0)
        if usage.get("cost") is not None:
            cost, source = float(usage["cost"]), "api"
        else:
            cost, source = self.pricing.cost(pt, ct), "pricing"
        return ChatResponse(
            text=text.strip(),
            latency_s=latency,
            model=str(data.get("model") or model),
            prompt_tokens=pt,
            completion_tokens=ct,
            cost_usd=cost,
            cost_source=source,
            raw=data,
        )


_KEY_CONCEPT_RE = re.compile(r"Key concept:\s*(.+?)(?:\.\s|\.$|$)", re.S)
_ADVERSARIAL_MARKERS = tuple(d.doc_text.split(".")[0] for d in ADVERSARIAL_DOCS)


class MockProvider:
    """Deterministic offline provider for tests and dry runs.

    It is intentionally crude and is *not* calibrated to reproduce any
    published number.  Generation copies the first ``Key concept`` sentence of
    the top context document (so a poisoned top document is copied faithfully,
    the failure mode L2 is designed to prevent); closed-book prompts get a
    placeholder answer.  The verifier/judge roles detect the adversarial marker
    sentences and the placeholder.
    """

    name = "mock"

    def __init__(self, model: str = "mock-model", latency_s: float = 0.0) -> None:
        self.model = model
        self.latency_s = latency_s
        self.calls = 0

    def chat(
        self,
        messages: Sequence[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        self.calls += 1
        user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        if json_mode and "verification judge" in user:
            text = self._verify(user)
        elif json_mode and "expert AI evaluator" in user:
            text = self._judge(user)
        else:
            text = self._generate(user)
        pt, ct = len(user.split()), len(text.split())
        return ChatResponse(text, self.latency_s, model or self.model, pt, ct, 0.0, "mock")

    def stream_chat(self, messages: Sequence[dict[str, str]], **kw: Any) -> Iterator[str]:
        resp = self.chat(messages, **kw)
        for i, word in enumerate(resp.text.split(" ")):
            yield word if i == 0 else " " + word
        return resp

    @staticmethod
    def _context_sources(prompt: str) -> list[tuple[str, str]]:
        """Sources inside the final ``Context:`` block only (few-shot examples are skipped)."""
        block = prompt.rsplit("Context:\n", 1)[-1]
        for stop in ("\n\nNegative Constraints", "\n\nQuestion:", "\n\nDraft Answer:", "\n\nModel Generated Response:"):
            block = block.split(stop, 1)[0]
        return re.findall(r"Source \[([^\]]+)\]: (.+)", block)

    def _top_context_sentence(self, prompt: str) -> str | None:
        sources = self._context_sources(prompt)
        if not sources:
            return None
        # Mimic the failure mode L2 guards against: if an adversarial document
        # is anywhere in the context, the mock "faithfully copies the lie".
        doc_text = next((t for i, t in sources if "POISON" in i), sources[0][1])
        kc = _KEY_CONCEPT_RE.search(doc_text)
        return (kc.group(1) if kc else doc_text.split(".")[0]).strip()

    def _generate(self, prompt: str) -> str:
        if not self._context_sources(prompt):
            if "Negative Constraints" in prompt:
                return "I do not know."
            q = re.search(r"Question: (.+)", prompt)
            return f"[unverified placeholder answer to: {q.group(1).strip() if q else 'question'}]"
        return self._top_context_sentence(prompt) or "I do not know."

    @staticmethod
    def _hallucinated(text: str) -> bool:
        return "[unverified placeholder" in text or any(mk in text for mk in _ADVERSARIAL_MARKERS)

    def _verify(self, prompt: str) -> str:
        draft = prompt.split("Draft Answer:", 1)[1].split("Your task:", 1)[0].strip()
        clean = None
        for sid, stext in self._context_sources(prompt):
            if "POISON" not in sid:
                kc = _KEY_CONCEPT_RE.search(stext)
                clean = (kc.group(1) if kc else stext.split(".")[0]).strip()
                break
        if self._hallucinated(draft):
            final = clean or "I do not know."
            return json.dumps({"claims": [{"claim": draft, "judgment": "CONTRADICTED", "evidence": [sid for sid, _ in self._context_sources(prompt)]}], "corrected_required": True, "final_answer": final})
        return json.dumps({"claims": [{"claim": draft, "judgment": "SUPPORTED", "evidence": [sid for sid, _ in self._context_sources(prompt)]}], "corrected_required": False, "final_answer": draft})

    def _judge(self, prompt: str) -> str:
        resp = prompt.split("Model Generated Response:", 1)[1].split("Your grading criteria", 1)[0].strip()
        if self._hallucinated(resp):
            return json.dumps({"helpfulness": 1, "hallucination": "Yes", "rationale": "contains an unsupported or adversarial claim"})
        if resp.lower().startswith("i do not know"):
            return json.dumps({"helpfulness": 3, "hallucination": "No", "rationale": "refusal"})
        return json.dumps({"helpfulness": 4, "hallucination": "No", "rationale": "grounded"})


def make_provider(
    provider: str = "openrouter",
    *,
    model: str,
    api_key: str | None = None,
    base_url: str | None = None,
    pricing: Pricing | None = None,
    timeout: float = 60.0,
) -> ChatProvider:
    """Build a provider by short name (``mock``, ``openrouter``, ``openai``, ``gemini``, ``groq``, ``ollama`` or ``custom``).

    The API key is read from the provider's conventional environment variable
    when not given.  Missing keys raise :class:`ProviderError` up front instead
    of failing on the first request.
    """
    if provider == "mock":
        return MockProvider(model=model)
    url = base_url or KNOWN_BASE_URLS.get(provider)
    if not url:
        raise ProviderError(f"unknown provider {provider!r}; pass --base-url for a custom endpoint")
    key = api_key or os.environ.get(API_KEY_ENV.get(provider, "LLM_API_KEY"))
    if not key and provider != "ollama":
        raise ProviderError(
            f"no API key for provider {provider!r}; set {API_KEY_ENV.get(provider, 'LLM_API_KEY')} or pass --api-key"
        )
    headers = {"HTTP-Referer": "https://github.com/rajeshgovindarajan/llm-layered-defense", "X-Title": "llm-layered-defense"} if provider == "openrouter" else None
    return OpenAICompatibleProvider(url, key, model, name=provider, pricing=pricing, timeout=timeout, extra_headers=headers)
