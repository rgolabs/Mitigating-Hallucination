"""Bring-your-own-LLM layer.

Any object with ``chat(messages, *, model, temperature, json_mode, max_tokens)
-> ChatResponse`` is an :class:`LLM`.  This module adds wrappers for plain
callables, the official ``openai`` and ``anthropic`` SDK clients and LangChain
chat models, plus :func:`resolve_llm` which accepts a ``"provider:model"``
string, a callable or an LLM instance.  Async is available for every LLM via
:func:`achat` (native ``achat`` when the object has one, otherwise a thread).
"""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Callable, Iterator, Sequence
from typing import Any, Protocol, runtime_checkable

from .providers import KNOWN_BASE_URLS, ChatResponse, MockProvider, OpenAICompatibleProvider, Pricing, ProviderError, make_provider

Messages = Sequence[dict[str, str]]


@runtime_checkable
class LLM(Protocol):
    name: str

    def chat(
        self,
        messages: Messages,
        *,
        model: str | None = None,
        temperature: float = 0.0,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> ChatResponse: ...


async def achat(llm: Any, messages: Messages, **kw: Any) -> ChatResponse:
    """Call ``llm.achat`` when available, else run ``llm.chat`` in a worker thread."""
    native = getattr(llm, "achat", None)
    if native is not None:
        return await native(messages, **kw)
    return await asyncio.to_thread(llm.chat, messages, **kw)


def _split_system(messages: Messages) -> tuple[str | None, list[dict[str, str]]]:
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system") or None
    rest = [m for m in messages if m["role"] != "system"]
    return system, rest


_JSON_HINT = "\n\nRespond with a single JSON object and nothing else."


def _with_json_hint(messages: Messages) -> list[dict[str, str]]:
    msgs = [dict(m) for m in messages]
    for m in reversed(msgs):
        if m["role"] == "user":
            m["content"] = m["content"].rstrip() + _JSON_HINT
            break
    return msgs


class CallableLLM:
    """Adapt a function ``fn(messages, **kwargs) -> str | ChatResponse`` (sync or async).

    ``kwargs`` (``model``, ``temperature``, ``json_mode``, ``max_tokens``) are
    passed only when ``fn`` accepts them, so ``lambda messages: ...`` works.
    """

    def __init__(self, fn: Callable[..., Any], *, name: str = "callable", model: str = "custom") -> None:
        self.fn = fn
        self.name = name
        self.model = model
        self._is_async = inspect.iscoroutinefunction(fn)
        try:
            params = inspect.signature(fn).parameters
        except (TypeError, ValueError):
            params = {}
        self._accepts_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
        self._named = {n for n, p in params.items() if p.kind in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)}

    def _kwargs(self, kw: dict[str, Any]) -> dict[str, Any]:
        if self._accepts_kwargs:
            return kw
        return {k: v for k, v in kw.items() if k in self._named}

    def _wrap(self, out: Any, started: float, model: str | None) -> ChatResponse:
        if isinstance(out, ChatResponse):
            return out
        return ChatResponse(str(out).strip(), time.perf_counter() - started, model or self.model, cost_source="none")

    def chat(self, messages: Messages, *, model: str | None = None, temperature: float = 0.0, json_mode: bool = False, max_tokens: int | None = None) -> ChatResponse:
        kw = self._kwargs({"model": model, "temperature": temperature, "json_mode": json_mode, "max_tokens": max_tokens})
        started = time.perf_counter()
        if self._is_async:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return self._wrap(asyncio.run(self.fn(list(messages), **kw)), started, model)
            raise RuntimeError("CallableLLM wraps an async function; use achat()/aanswer() inside a running event loop")
        return self._wrap(self.fn(list(messages), **kw), started, model)

    async def achat(self, messages: Messages, *, model: str | None = None, temperature: float = 0.0, json_mode: bool = False, max_tokens: int | None = None) -> ChatResponse:
        kw = self._kwargs({"model": model, "temperature": temperature, "json_mode": json_mode, "max_tokens": max_tokens})
        started = time.perf_counter()
        if self._is_async:
            out = await self.fn(list(messages), **kw)
        else:
            out = await asyncio.to_thread(self.fn, list(messages), **kw)
        return self._wrap(out, started, model)


class OpenAILLM:
    """Wrap an ``openai.OpenAI`` (or any client exposing ``chat.completions.create``)."""

    name = "openai-sdk"

    def __init__(self, client: Any, model: str, *, pricing: Pricing | None = None) -> None:
        self.client = client
        self.model = model
        self.pricing = pricing or Pricing()

    def chat(self, messages: Messages, *, model: str | None = None, temperature: float = 0.0, json_mode: bool = False, max_tokens: int | None = None) -> ChatResponse:
        kw: dict[str, Any] = {"model": model or self.model, "messages": list(messages), "temperature": temperature}
        if json_mode:
            kw["response_format"] = {"type": "json_object"}
        if max_tokens:
            kw["max_tokens"] = max_tokens
        started = time.perf_counter()
        res = self.client.chat.completions.create(**kw)
        latency = time.perf_counter() - started
        text = (res.choices[0].message.content or "").strip()
        usage = getattr(res, "usage", None)
        pt, ct = int(getattr(usage, "prompt_tokens", 0) or 0), int(getattr(usage, "completion_tokens", 0) or 0)
        return ChatResponse(text, latency, str(getattr(res, "model", kw["model"])), pt, ct, self.pricing.cost(pt, ct), "pricing")


class AnthropicLLM:
    """Wrap an ``anthropic.Anthropic`` client (``messages.create``)."""

    name = "anthropic-sdk"

    def __init__(self, client: Any, model: str, *, pricing: Pricing | None = None, default_max_tokens: int = 1024) -> None:
        self.client = client
        self.model = model
        self.pricing = pricing or Pricing()
        self.default_max_tokens = default_max_tokens

    def chat(self, messages: Messages, *, model: str | None = None, temperature: float = 0.0, json_mode: bool = False, max_tokens: int | None = None) -> ChatResponse:
        system, rest = _split_system(_with_json_hint(messages) if json_mode else messages)
        kw: dict[str, Any] = {"model": model or self.model, "messages": rest, "temperature": temperature, "max_tokens": max_tokens or self.default_max_tokens}
        if system:
            kw["system"] = system
        started = time.perf_counter()
        res = self.client.messages.create(**kw)
        latency = time.perf_counter() - started
        text = "".join(getattr(b, "text", "") for b in res.content).strip()
        usage = getattr(res, "usage", None)
        pt, ct = int(getattr(usage, "input_tokens", 0) or 0), int(getattr(usage, "output_tokens", 0) or 0)
        return ChatResponse(text, latency, str(getattr(res, "model", kw["model"])), pt, ct, self.pricing.cost(pt, ct), "pricing")


class LangChainLLM:
    """Wrap a LangChain ``BaseChatModel`` (uses ``invoke``/``ainvoke``)."""

    name = "langchain"

    def __init__(self, chat_model: Any, *, model: str | None = None) -> None:
        self.chat_model = chat_model
        self.model = model or str(getattr(chat_model, "model_name", None) or getattr(chat_model, "model", "langchain"))

    @staticmethod
    def _convert(messages: Messages, json_mode: bool) -> list[tuple[str, str]]:
        role = {"user": "human", "assistant": "ai", "system": "system"}
        return [(role.get(m["role"], m["role"]), m["content"]) for m in (_with_json_hint(messages) if json_mode else messages)]

    @staticmethod
    def _wrap(res: Any, started: float, model: str) -> ChatResponse:
        content = res.content if isinstance(res.content, str) else "".join(str(getattr(p, "text", p)) for p in res.content)
        um = getattr(res, "usage_metadata", None) or {}
        pt, ct = int(um.get("input_tokens", 0) or 0), int(um.get("output_tokens", 0) or 0)
        return ChatResponse(content.strip(), time.perf_counter() - started, model, pt, ct, 0.0, "none")

    def chat(self, messages: Messages, *, model: str | None = None, temperature: float = 0.0, json_mode: bool = False, max_tokens: int | None = None) -> ChatResponse:
        started = time.perf_counter()
        return self._wrap(self.chat_model.invoke(self._convert(messages, json_mode)), started, model or self.model)

    async def achat(self, messages: Messages, *, model: str | None = None, temperature: float = 0.0, json_mode: bool = False, max_tokens: int | None = None) -> ChatResponse:
        started = time.perf_counter()
        return self._wrap(await self.chat_model.ainvoke(self._convert(messages, json_mode)), started, model or self.model)


def resolve_llm(spec: Any, *, api_key: str | None = None, base_url: str | None = None, pricing: Pricing | None = None) -> Any:
    """Turn ``"provider:model"``, a callable or an LLM object into an LLM.

    Provider names: ``openrouter``, ``openai``, ``gemini``, ``groq``, ``ollama``,
    ``mock``, or ``custom`` (needs ``base_url``).  Examples::

        resolve_llm("openrouter:google/gemini-2.5-flash")
        resolve_llm("custom:my-model", base_url="http://127.0.0.1:8787/v1")
        resolve_llm(lambda messages: my_client(messages))
    """
    if isinstance(spec, str):
        if ":" not in spec:
            raise ProviderError(f"LLM spec {spec!r} must look like 'provider:model' (providers: {', '.join([*KNOWN_BASE_URLS, 'custom', 'mock'])})")
        provider, model = spec.split(":", 1)
        return make_provider(provider, model=model, api_key=api_key, base_url=base_url, pricing=pricing)
    if hasattr(spec, "chat") and callable(spec.chat):
        return spec
    if callable(spec):
        return CallableLLM(spec)
    raise TypeError(f"cannot use {type(spec).__name__} as an LLM; pass 'provider:model', a callable or an object with .chat()")


def stream_chat(llm: Any, messages: Messages, **kw: Any) -> Iterator[str]:
    """Yield text chunks from ``llm.stream_chat`` if it exists, else one chunk from ``chat``.

    The generator's return value is the final :class:`ChatResponse`.
    """
    native = getattr(llm, "stream_chat", None)
    if native is not None:
        return native(messages, **kw)

    def _single() -> Iterator[str]:
        resp = llm.chat(messages, **kw)
        yield resp.text
        return resp

    return _single()


__all__ = ["LLM", "CallableLLM", "OpenAILLM", "AnthropicLLM", "LangChainLLM", "MockProvider", "OpenAICompatibleProvider", "achat", "resolve_llm", "stream_chat"]
