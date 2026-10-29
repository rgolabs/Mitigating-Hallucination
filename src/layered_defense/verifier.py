"""Security Layer 1: claim-level verification of a draft answer by a second model."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .corpus import Document
from .prompts import render_context
from .providers import ChatResponse

JUDGMENTS = ("SUPPORTED", "CONTRADICTED", "UNVERIFIABLE")

VERIFIER_PROMPT = """You are a verification judge. Analyze the draft answer to the question using the provided context.

Question: {question}
Context:
{context}

Draft Answer:
{draft}

Your task:
1. Extract all atomic factual claims from the draft answer.
2. For each claim, determine if it is:
   - SUPPORTED: explicitly stated in or directly inferred from the context{parametric_support}.
   - CONTRADICTED: directly contradicted by the context{parametric_contra}.
   - UNVERIFIABLE: not mentioned or not checkable from the context.
   List the source ids (the text inside Source [...]) that support or contradict each claim.
3. If any claim is CONTRADICTED or UNVERIFIABLE, write a revised answer that removes or corrects those claims while keeping supported content unchanged. If nothing supported remains, answer exactly "I do not know."
4. If all claims are SUPPORTED, accept the draft answer unchanged.

Output only this JSON object:
{{"claims": [{{"claim": "...", "judgment": "SUPPORTED|CONTRADICTED|UNVERIFIABLE", "evidence": ["source id", "..."]}}], "corrected_required": true|false, "final_answer": "..."}}
"""


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object from model output, tolerating code fences and prose."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fence:
        text = fence.group(1)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1:
            raise
        obj = json.loads(text[start : end + 1])
    if not isinstance(obj, dict):
        raise ValueError("expected a JSON object")
    return obj


@dataclass
class Claim:
    text: str
    judgment: str
    evidence: list[str] = field(default_factory=list)

    @property
    def supported(self) -> bool:
        return self.judgment == "SUPPORTED"


@dataclass
class Verdict:
    """Outcome of verifying one draft answer."""

    claims: list[Claim]
    corrected_required: bool
    final_answer: str
    parse_ok: bool
    response: ChatResponse

    @property
    def contradicted(self) -> list[Claim]:
        return [c for c in self.claims if c.judgment == "CONTRADICTED"]

    @property
    def unverifiable(self) -> list[Claim]:
        return [c for c in self.claims if c.judgment == "UNVERIFIABLE"]

    @property
    def all_supported(self) -> bool:
        return self.parse_ok and all(c.supported for c in self.claims)

    def to_dict(self) -> dict[str, Any]:
        return {
            "claims": [{"claim": c.text, "judgment": c.judgment, "evidence": c.evidence} for c in self.claims],
            "corrected_required": self.corrected_required,
            "final_answer": self.final_answer,
            "parse_ok": self.parse_ok,
        }


def build_verifier_prompt(question: str, draft: str, context: Sequence[Document], *, allow_parametric: bool = False) -> str:
    """``allow_parametric`` lets well-established facts count as evidence when no context is given."""
    parametric = allow_parametric or not context
    return VERIFIER_PROMPT.format(
        question=question,
        context=render_context(context),
        draft=draft,
        parametric_support=" (or, if no context is available, a well-established fact)" if parametric else "",
        parametric_contra=" (or by well-established facts if no context is available)" if parametric else "",
    )


def _parse_verdict(text: str, draft: str, response: ChatResponse) -> Verdict:
    try:
        obj = parse_json_object(text)
    except (ValueError, json.JSONDecodeError):
        # An unparsable verdict means "could not verify": the draft passes
        # through unchanged and the raw output is kept for audit.
        return Verdict([Claim("<unparsable verifier output>", "UNVERIFIABLE")], False, draft, False, response)
    claims = []
    for c in obj.get("claims", []) or []:
        if not isinstance(c, dict):
            continue
        judgment = str(c.get("judgment", "UNVERIFIABLE")).strip().upper()
        if judgment not in JUDGMENTS:
            judgment = "UNVERIFIABLE"
        ev = c.get("evidence") or []
        claims.append(Claim(str(c.get("claim", "")), judgment, [str(e) for e in ev] if isinstance(ev, list) else [str(ev)]))
    final = str(obj.get("final_answer") or draft).strip()
    flagged = bool(obj.get("corrected_required", False)) or any(not c.supported for c in claims)
    return Verdict(claims, flagged, final, True, response)


def verify_claims(
    llm: Any,
    question: str,
    draft: str,
    context: Sequence[Document],
    *,
    model: str | None = None,
    max_tokens: int = 1024,
    allow_parametric: bool = False,
) -> Verdict:
    prompt = build_verifier_prompt(question, draft, context, allow_parametric=allow_parametric)
    resp = llm.chat([{"role": "user", "content": prompt}], model=model, temperature=0.0, json_mode=True, max_tokens=max_tokens)
    return _parse_verdict(resp.text, draft, resp)


async def averify_claims(
    llm: Any,
    question: str,
    draft: str,
    context: Sequence[Document],
    *,
    model: str | None = None,
    max_tokens: int = 1024,
    allow_parametric: bool = False,
) -> Verdict:
    from .llm import achat

    prompt = build_verifier_prompt(question, draft, context, allow_parametric=allow_parametric)
    resp = await achat(llm, [{"role": "user", "content": prompt}], model=model, temperature=0.0, json_mode=True, max_tokens=max_tokens)
    return _parse_verdict(resp.text, draft, resp)


__all__ = ["Claim", "Verdict", "VERIFIER_PROMPT", "build_verifier_prompt", "parse_json_object", "verify_claims", "averify_claims", "asyncio"]
