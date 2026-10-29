"""LangChain / LangGraph adapter: ``as_runnable(guard)`` returns a Runnable.

Input: ``str`` or ``{"question": str, "context": [...], "domain": str, "history": [...]}``.
Output: :class:`layered_defense.GuardedAnswer`.  Requires ``langchain-core``.
"""

from __future__ import annotations

from typing import Any

from ..guard import GuardedAnswer, LayeredDefense


def _unpack(inp: Any) -> dict[str, Any]:
    if isinstance(inp, str):
        return {"question": inp}
    if isinstance(inp, dict) and "question" in inp:
        return {k: inp.get(k) for k in ("question", "context", "domain", "history") if k in inp}
    raise TypeError("input must be a question string or a dict with a 'question' key")


def as_runnable(guard: LayeredDefense) -> Any:
    try:
        from langchain_core.runnables import RunnableLambda
    except ImportError as err:  # pragma: no cover
        raise ImportError("install langchain-core to use as_runnable()") from err

    def _sync(inp: Any) -> GuardedAnswer:
        return guard.answer(**_unpack(inp))

    async def _async(inp: Any) -> GuardedAnswer:
        return await guard.aanswer(**_unpack(inp))

    return RunnableLambda(_sync, afunc=_async, name="layered_defense")
