"""The importable runtime guard: wrap any LLM call with the four defense layers.

Typical use inside an agent::

    from layered_defense import LayeredDefense

    guard = LayeredDefense("openrouter:google/gemini-2.5-flash", retriever=my_retriever)
    out = guard.answer("What does our refund policy say?", context=tool_results)
    out.text        # grounded answer, rewritten/refused/flagged per policy
    out.claims      # claim-level audit trail from the verifier
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from .corpus import Document, MetadataPolicy
from .llm import achat, resolve_llm, stream_chat
from .prompts import ALIGNMENT_PERSONAS, PROMPT_CONFIGS, format_prompt, is_refusal
from .providers import ChatResponse
from .retrieval import CallableRetriever, apply_policy, aretrieve, as_documents
from .verifier import Claim, Verdict, averify_claims, verify_claims

Action = Literal["rewrite", "refuse", "flag", "allow"]


@dataclass(frozen=True)
class GuardPolicy:
    """What the guard does with the verifier's verdict, and how much it spends.

    * ``verify``: run the L1 verifier at all.  ``verify_when(question, docs)``
      can switch it off for low-risk calls (risk tiering).
    * ``on_contradicted`` / ``on_unverifiable``: ``rewrite`` (use the verifier's
      corrected answer), ``refuse`` (return ``refusal_text``), ``flag`` (keep
      the draft, mark it) or ``allow`` (ignore that class of claim).  Closed-book
      agents that rely on parametric knowledge usually want
      ``on_unverifiable="flag"`` or ``"allow"`` plus ``allow_parametric=True``.
    * ``prompt`` / ``alignment``: Layer 3 condition and Layer 4 persona.
    """

    verify: bool = True
    verify_when: Callable[[str, list[Document]], bool] | None = None
    on_contradicted: Action = "rewrite"
    on_unverifiable: Action = "rewrite"
    allow_parametric: bool = False
    refusal_text: str = "I do not know."
    prompt: str = "full"
    alignment: str = "aligned"
    top_k: int = 5
    max_context_chars: int | None = 24_000
    temperature: float | None = None
    max_tokens: int = 1024
    verifier_max_tokens: int = 2048

    def __post_init__(self) -> None:
        if self.prompt not in PROMPT_CONFIGS:
            raise ValueError(f"prompt must be one of {PROMPT_CONFIGS}")
        if self.alignment not in ALIGNMENT_PERSONAS:
            raise ValueError(f"alignment must be one of {tuple(ALIGNMENT_PERSONAS)}")
        for a in (self.on_contradicted, self.on_unverifiable):
            if a not in ("rewrite", "refuse", "flag", "allow"):
                raise ValueError(f"unknown action {a!r}")


@dataclass
class GuardEvent:
    type: str  # retrieve | generate | token | verify | decision
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class GuardedAnswer:
    question: str
    text: str
    draft: str
    context: list[Document]
    verdict: Verdict | None
    action: str  # accepted | rewritten | refused | flagged | unverified
    generation: ChatResponse
    prompt: str
    latency_s: float

    @property
    def claims(self) -> list[Claim]:
        return self.verdict.claims if self.verdict else []

    @property
    def corrected(self) -> bool:
        return self.action == "rewritten"

    @property
    def refused(self) -> bool:
        return self.action == "refused" or is_refusal(self.text)

    @property
    def flagged(self) -> bool:
        return self.action == "flagged"

    @property
    def grounded(self) -> bool:
        """True when the verifier ran, parsed, and found every claim supported (or the answer was rewritten/refused)."""
        return self.verdict is not None and self.verdict.parse_ok and self.action in ("accepted", "rewritten", "refused")

    @property
    def cost_usd(self) -> float:
        return self.generation.cost_usd + (self.verdict.response.cost_usd if self.verdict else 0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "text": self.text,
            "draft": self.draft,
            "action": self.action,
            "grounded": self.grounded,
            "context_ids": [d.doc_id for d in self.context],
            "verdict": self.verdict.to_dict() if self.verdict else None,
            "latency_s": round(self.latency_s, 4),
            "cost_usd": self.cost_usd,
            "model": self.generation.model,
        }


class LayeredDefense:
    """Compose retrieval, prompt hardening, generation and verification around any LLM.

    Parameters
    ----------
    llm:
        ``"provider:model"``, a callable ``fn(messages, **kw) -> str`` or any
        object with ``.chat()`` (see :mod:`layered_defense.llm`).
    verifier:
        Same forms; defaults to ``llm``.  Use a model at least as strong as the generator.
    retriever:
        Object with ``retrieve(query, k=, domain=)`` or a callable ``fn(query, k)``.
        Optional: agents can pass tool outputs via ``context=`` instead.
    metadata_policy:
        Applied to *all* context (retrieved and supplied) when given.  Leave
        ``None`` when supplying tool outputs that carry no metadata.
    system_prompt / examples / negative_constraints:
        Compose with (not replace) the layer-3 prompt.
    on_event:
        Callback receiving :class:`GuardEvent` for tracing/observability.
    """

    def __init__(
        self,
        llm: Any,
        *,
        verifier: Any = None,
        retriever: Any = None,
        policy: GuardPolicy | None = None,
        metadata_policy: MetadataPolicy | None = None,
        system_prompt: str | None = None,
        instruction: str | None = None,
        examples: str | None = None,
        negative_constraints: Sequence[str] | None = None,
        generator_model: str | None = None,
        verifier_model: str | None = None,
        on_event: Callable[[GuardEvent], None] | None = None,
    ) -> None:
        self.llm = resolve_llm(llm)
        self.verifier = resolve_llm(verifier) if verifier is not None else self.llm
        if retriever is not None and not hasattr(retriever, "retrieve") and callable(retriever):
            retriever = CallableRetriever(retriever)
        self.retriever = retriever
        self.policy = policy or GuardPolicy()
        self.metadata_policy = metadata_policy
        self.system_prompt = system_prompt
        self.instruction = instruction
        self.examples = examples
        self.negative_constraints = negative_constraints
        self.generator_model = generator_model
        self.verifier_model = verifier_model
        self.on_event = on_event

    # ------------------------------------------------------------ helpers
    def _emit(self, type_: str, **data: Any) -> None:
        if self.on_event:
            self.on_event(GuardEvent(type_, data))

    def _finish_context(self, docs: list[Document], domain: str | None) -> list[Document]:
        docs = apply_policy(docs, self.metadata_policy, domain)
        if self.policy.max_context_chars is not None:
            kept, used = [], 0
            for d in docs:
                used += len(d.doc_text)
                if used > self.policy.max_context_chars:
                    break
                kept.append(d)
            docs = kept
        self._emit("retrieve", doc_ids=[d.doc_id for d in docs], adversarial=[d.doc_id for d in docs if d.adversarial])
        return docs

    def build_context(self, question: str, context: Iterable[Any] | str | None = None, domain: str | None = None) -> list[Document]:
        docs = as_documents(context)
        if self.retriever is not None:
            docs += self.retriever.retrieve(question, k=self.policy.top_k, domain=domain)
        return self._finish_context(docs, domain)

    async def abuild_context(self, question: str, context: Iterable[Any] | str | None = None, domain: str | None = None) -> list[Document]:
        docs = as_documents(context)
        if self.retriever is not None:
            docs += await aretrieve(self.retriever, question, k=self.policy.top_k, domain=domain)
        return self._finish_context(docs, domain)

    def build_messages(self, question: str, docs: Sequence[Document], history: Sequence[dict[str, str]] | None = None) -> tuple[list[dict[str, str]], str]:
        persona = ALIGNMENT_PERSONAS[self.policy.alignment]
        system = persona.system_prompt + (f"\n\n{self.system_prompt}" if self.system_prompt else "")
        prompt = format_prompt(question, docs, self.policy.prompt, instruction=self.instruction, examples=self.examples, negative_constraints=self.negative_constraints)
        messages = [{"role": "system", "content": system}, *[dict(m) for m in (history or []) if m.get("role") in ("user", "assistant")], {"role": "user", "content": prompt}]
        return messages, prompt

    def _gen_kwargs(self) -> dict[str, Any]:
        persona = ALIGNMENT_PERSONAS[self.policy.alignment]
        temp = persona.temperature if self.policy.temperature is None else self.policy.temperature
        return {"model": self.generator_model, "temperature": temp, "max_tokens": self.policy.max_tokens}

    def _should_verify(self, question: str, docs: list[Document]) -> bool:
        if not self.policy.verify:
            return False
        return self.policy.verify_when(question, docs) if self.policy.verify_when else True

    def _decide(self, draft: str, verdict: Verdict | None) -> tuple[str, str]:
        """Apply the policy to the verdict; returns (text, action)."""
        if verdict is None:
            return draft, "unverified"
        if not verdict.parse_ok:
            return draft, "unverified"
        actions: list[str] = []
        if verdict.contradicted:
            actions.append(self.policy.on_contradicted)
        if verdict.unverifiable:
            actions.append(self.policy.on_unverifiable)
        # Also honour a verifier that asked for a rewrite without labelling a claim.
        if not actions and verdict.corrected_required and verdict.final_answer != draft:
            actions.append(self.policy.on_contradicted)
        actions = [a for a in actions if a != "allow"]
        if not actions:
            return draft, "accepted"
        if "refuse" in actions:
            return self.policy.refusal_text, "refused"
        if "rewrite" in actions:
            final = verdict.final_answer or self.policy.refusal_text
            return (final, "rewritten") if final != draft else (draft, "accepted")
        return draft, "flagged"

    # ------------------------------------------------------------ sync API
    def answer(self, question: str, *, context: Iterable[Any] | str | None = None, domain: str | None = None, history: Sequence[dict[str, str]] | None = None) -> GuardedAnswer:
        started = time.perf_counter()
        docs = self.build_context(question, context, domain)
        messages, prompt = self.build_messages(question, docs, history)
        gen = self.llm.chat(messages, **self._gen_kwargs())
        self._emit("generate", model=gen.model, latency_s=gen.latency_s, tokens=gen.prompt_tokens + gen.completion_tokens)
        verdict = None
        if self._should_verify(question, docs):
            verdict = verify_claims(self.verifier, question, gen.text, docs, model=self.verifier_model, max_tokens=self.policy.verifier_max_tokens, allow_parametric=self.policy.allow_parametric)
            self._emit("verify", **verdict.to_dict())
        text, action = self._decide(gen.text, verdict)
        self._emit("decision", action=action)
        return GuardedAnswer(question, text, gen.text, docs, verdict, action, gen, prompt, time.perf_counter() - started)

    def verify(self, question: str, answer: str, *, context: Iterable[Any] | str | None = None, domain: str | None = None) -> Verdict:
        """Verify an answer produced elsewhere (e.g. by an agent step) against context."""
        docs = self.build_context(question, context, domain)
        return verify_claims(self.verifier, question, answer, docs, model=self.verifier_model, max_tokens=self.policy.verifier_max_tokens, allow_parametric=self.policy.allow_parametric)

    def stream(self, question: str, *, context: Iterable[Any] | str | None = None, domain: str | None = None, history: Sequence[dict[str, str]] | None = None) -> Iterator[GuardEvent]:
        """Stream the draft as ``token`` events, then ``verify``/``decision`` and a final ``answer`` event.

        Consumers that show tokens as they arrive must be prepared to replace
        the shown draft with ``answer.data["text"]`` when the action is
        ``rewritten`` or ``refused``.
        """
        started = time.perf_counter()
        docs = self.build_context(question, context, domain)
        messages, prompt = self.build_messages(question, docs, history)
        gen_iter = stream_chat(self.llm, messages, **self._gen_kwargs())
        parts: list[str] = []
        response: ChatResponse | None = None
        try:
            while True:
                chunk = next(gen_iter)
                parts.append(chunk)
                yield GuardEvent("token", {"text": chunk})
        except StopIteration as stop:
            response = stop.value
        draft = "".join(parts).strip()
        gen = response or ChatResponse(draft, time.perf_counter() - started, str(self.generator_model or getattr(self.llm, "model", "")), cost_source="none")
        verdict = None
        if self._should_verify(question, docs):
            verdict = verify_claims(self.verifier, question, draft, docs, model=self.verifier_model, max_tokens=self.policy.verifier_max_tokens, allow_parametric=self.policy.allow_parametric)
            yield GuardEvent("verify", verdict.to_dict())
        text, action = self._decide(draft, verdict)
        yield GuardEvent("decision", {"action": action})
        result = GuardedAnswer(question, text, draft, docs, verdict, action, gen, prompt, time.perf_counter() - started)
        yield GuardEvent("answer", {"text": text, "result": result})

    # ----------------------------------------------------------- async API
    async def aanswer(self, question: str, *, context: Iterable[Any] | str | None = None, domain: str | None = None, history: Sequence[dict[str, str]] | None = None) -> GuardedAnswer:
        started = time.perf_counter()
        docs = await self.abuild_context(question, context, domain)
        messages, prompt = self.build_messages(question, docs, history)
        gen = await achat(self.llm, messages, **self._gen_kwargs())
        self._emit("generate", model=gen.model, latency_s=gen.latency_s, tokens=gen.prompt_tokens + gen.completion_tokens)
        verdict = None
        if self._should_verify(question, docs):
            verdict = await averify_claims(self.verifier, question, gen.text, docs, model=self.verifier_model, max_tokens=self.policy.verifier_max_tokens, allow_parametric=self.policy.allow_parametric)
            self._emit("verify", **verdict.to_dict())
        text, action = self._decide(gen.text, verdict)
        self._emit("decision", action=action)
        return GuardedAnswer(question, text, gen.text, docs, verdict, action, gen, prompt, time.perf_counter() - started)

    async def averify(self, question: str, answer: str, *, context: Iterable[Any] | str | None = None, domain: str | None = None) -> Verdict:
        docs = await self.abuild_context(question, context, domain)
        return await averify_claims(self.verifier, question, answer, docs, model=self.verifier_model, max_tokens=self.policy.verifier_max_tokens, allow_parametric=self.policy.allow_parametric)

    def __call__(self, question: str, **kw: Any) -> GuardedAnswer:
        return self.answer(question, **kw)


Guard = LayeredDefense

__all__ = ["LayeredDefense", "Guard", "GuardPolicy", "GuardedAnswer", "GuardEvent", "asyncio"]
