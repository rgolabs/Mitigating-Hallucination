"""Expose the guard as plain tool functions for OpenAI / Anthropic / any tool-calling agent."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..guard import LayeredDefense

_VERIFY_PARAMS = {
    "type": "object",
    "properties": {
        "question": {"type": "string", "description": "The user's question the answer responds to."},
        "answer": {"type": "string", "description": "The draft answer to verify."},
        "context": {"type": "array", "items": {"type": "string"}, "description": "Evidence snippets (tool outputs, retrieved passages)."},
    },
    "required": ["question", "answer"],
}

OPENAI_VERIFY_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "verify_answer",
        "description": "Check a draft answer claim by claim against evidence; returns judgments and a corrected answer.",
        "parameters": _VERIFY_PARAMS,
    },
}

ANTHROPIC_VERIFY_TOOL: dict[str, Any] = {
    "name": "verify_answer",
    "description": "Check a draft answer claim by claim against evidence; returns judgments and a corrected answer.",
    "input_schema": _VERIFY_PARAMS,
}


def make_verify_tool(guard: LayeredDefense) -> Callable[..., dict[str, Any]]:
    """Return ``verify_answer(question, answer, context=None) -> dict`` bound to ``guard``."""

    def verify_answer(question: str, answer: str, context: list[str] | None = None) -> dict[str, Any]:
        return guard.verify(question, answer, context=context).to_dict()

    verify_answer.__doc__ = OPENAI_VERIFY_TOOL["function"]["description"]
    return verify_answer


def make_answer_tool(guard: LayeredDefense) -> Callable[..., dict[str, Any]]:
    """Return ``grounded_answer(question, context=None) -> dict`` bound to ``guard``."""

    def grounded_answer(question: str, context: list[str] | None = None) -> dict[str, Any]:
        return guard.answer(question, context=context).to_dict()

    grounded_answer.__doc__ = "Answer a question using only the supplied evidence, with claim-level verification."
    return grounded_answer
