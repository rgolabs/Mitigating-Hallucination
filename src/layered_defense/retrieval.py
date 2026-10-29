"""Security Layer 2 as a pluggable protocol: bring your own retriever or context."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable
from typing import Any, Protocol, runtime_checkable

from .corpus import DocCorpus, Document, MetadataPolicy

_TEXT_KEYS = ("doc_text", "text", "content", "page_content", "body", "snippet")
_ID_KEYS = ("doc_id", "id", "source_id", "url", "name")
_DOMAIN_KEYS = ("doc_domain", "domain", "category")
_TITLE_KEYS = ("doc_title", "title")
_SOURCE_KEYS = ("source_name", "source", "author", "tool")
_YEAR_KEYS = ("publication_year", "year", "date", "published")
_QUALITY_KEYS = ("quality_score", "trust", "trust_score", "score", "quality")


def _first(d: dict[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return default


def _year(v: Any) -> int:
    if v in (None, ""):
        return 0
    try:
        return int(float(v))
    except (TypeError, ValueError):
        digits = "".join(ch for ch in str(v)[:4] if ch.isdigit())
        return int(digits) if len(digits) == 4 else 0


def as_documents(items: Iterable[Any] | str | None, *, prefix: str = "ctx") -> list[Document]:
    """Coerce strings, dicts, LangChain documents or :class:`Document` into documents.

    Tool outputs can be passed as plain strings; they become ``Source [ctx-1]``
    etc.  Dicts may use common keys (``text``/``content``/``page_content``,
    ``id``/``url``, ``source``, ``date``/``year``, ``trust``/``score``); unknown
    keys are kept in ``Document.metadata``.
    """
    if items is None:
        return []
    if isinstance(items, str):
        items = [items]
    docs: list[Document] = []
    for i, item in enumerate(items, 1):
        if isinstance(item, Document):
            docs.append(item)
            continue
        if isinstance(item, str):
            docs.append(Document(f"{prefix}-{i}", item))
            continue
        if hasattr(item, "page_content"):  # LangChain / LlamaIndex style objects
            meta = dict(getattr(item, "metadata", {}) or {})
            item = {"page_content": item.page_content, **meta}
        if not isinstance(item, dict):
            docs.append(Document(f"{prefix}-{i}", str(item)))
            continue
        text = _first(item, _TEXT_KEYS)
        if text is None:
            raise ValueError(f"context item {i} has no text field (expected one of {_TEXT_KEYS})")
        known = set(_TEXT_KEYS) | set(_ID_KEYS) | set(_DOMAIN_KEYS) | set(_TITLE_KEYS) | set(_SOURCE_KEYS) | set(_YEAR_KEYS) | set(_QUALITY_KEYS)
        docs.append(
            Document(
                doc_id=str(_first(item, _ID_KEYS, f"{prefix}-{i}")),
                doc_text=str(text),
                doc_domain=str(_first(item, _DOMAIN_KEYS, "")),
                doc_title=str(_first(item, _TITLE_KEYS, "")),
                source_name=str(_first(item, _SOURCE_KEYS, "")),
                publication_year=_year(_first(item, _YEAR_KEYS)),
                quality_score=float(_first(item, _QUALITY_KEYS, 0.0) or 0.0),
                adversarial=bool(item.get("adversarial", False)),
                metadata={k: v for k, v in item.items() if k not in known and k != "adversarial"},
            )
        )
    return docs


@runtime_checkable
class Retriever(Protocol):
    """Anything that returns candidate documents for a query."""

    def retrieve(self, query: str, *, k: int = 5, domain: str | None = None) -> list[Document]: ...


async def aretrieve(retriever: Any, query: str, *, k: int = 5, domain: str | None = None) -> list[Document]:
    native = getattr(retriever, "aretrieve", None)
    if native is not None:
        return await native(query, k=k, domain=domain)
    return await asyncio.to_thread(retriever.retrieve, query, k=k, domain=domain)


class CallableRetriever:
    """Adapt ``fn(query, k) -> iterable`` (sync or async); results pass through :func:`as_documents`."""

    def __init__(self, fn: Callable[..., Any], *, name: str = "callable") -> None:
        self.fn = fn
        self.name = name
        self._is_async = asyncio.iscoroutinefunction(fn)

    def retrieve(self, query: str, *, k: int = 5, domain: str | None = None) -> list[Document]:
        if self._is_async:
            return asyncio.run(self.aretrieve(query, k=k, domain=domain))
        return as_documents(self.fn(query, k), prefix="doc")

    async def aretrieve(self, query: str, *, k: int = 5, domain: str | None = None) -> list[Document]:
        out = await self.fn(query, k) if self._is_async else await asyncio.to_thread(self.fn, query, k)
        return as_documents(out, prefix="doc")


class LexicalRetriever:
    """The bundled corpus retriever with the paper's metadata filter (Layer 2)."""

    name = "lexical"

    def __init__(self, corpus: DocCorpus, *, policy: MetadataPolicy | None | str = "default", candidate_k: int = 20, include_adversarial: bool = True) -> None:
        self.corpus = corpus
        self.policy = MetadataPolicy() if policy == "default" else policy
        self.candidate_k = candidate_k
        self.include_adversarial = include_adversarial

    def retrieve(self, query: str, *, k: int = 5, domain: str | None = None) -> list[Document]:
        return self.corpus.retrieve(query, domain, top_k=k, policy=self.policy, include_adversarial=self.include_adversarial, candidate_k=self.candidate_k)


def apply_policy(docs: list[Document], policy: MetadataPolicy | None, domain: str | None) -> list[Document]:
    return docs if policy is None else [d for d in docs if policy.accepts(d, domain)]


__all__ = ["Retriever", "CallableRetriever", "LexicalRetriever", "as_documents", "apply_policy", "aretrieve"]
