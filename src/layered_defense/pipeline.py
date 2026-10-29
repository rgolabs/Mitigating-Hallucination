"""The layered defense pipeline P = (R, F, G, V)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .corpus import DocCorpus, Document, MetadataPolicy
from .prompts import ALIGNMENT_PERSONAS, format_prompt, is_refusal
from .providers import ChatProvider, ChatResponse
from .verifier import VERIFIER_PROMPT, parse_json_object, verify_claims  # noqa: F401  (re-exported for compatibility)

RAG_MODES = ("none", "vanilla", "poisoned", "filtered")


@dataclass(frozen=True)
class LayerConfig:
    """Which layers are active for one pipeline configuration.

    * ``rag``: ``none`` (closed book), ``vanilla`` (top-k, clean index),
      ``poisoned`` (top-k over an index with adversarial documents, no filter)
      or ``filtered`` (poisoned index + metadata policy) -- Layer 2.
    * ``prompt``: one of :data:`layered_defense.prompts.PROMPT_CONFIGS` -- Layer 3.
    * ``alignment``: key of :data:`layered_defense.prompts.ALIGNMENT_PERSONAS` -- Layer 4.
    * ``verifier``: run the LLM-as-a-Judge rewrite step -- Layer 1.
    """

    name: str
    rag: str = "none"
    prompt: str = "bare"
    alignment: str = "base"
    verifier: bool = False
    row: int | None = None

    def __post_init__(self) -> None:
        if self.rag not in RAG_MODES:
            raise ValueError(f"rag must be one of {RAG_MODES}, got {self.rag!r}")
        if self.alignment not in ALIGNMENT_PERSONAS:
            raise ValueError(f"alignment must be one of {tuple(ALIGNMENT_PERSONAS)}, got {self.alignment!r}")

    def as_dict(self) -> dict[str, Any]:
        return {"row": self.row, "name": self.name, "rag": self.rag, "prompt": self.prompt, "alignment": self.alignment, "verifier": self.verifier}


@dataclass
class PipelineResult:
    question: str
    domain: str
    config: LayerConfig
    context: list[Document]
    prompt: str
    draft_answer: str
    final_answer: str
    corrected: bool
    verifier_claims: list[dict[str, str]]
    generation: ChatResponse
    verification: ChatResponse | None = None

    @property
    def latency_s(self) -> float:
        return self.generation.latency_s + (self.verification.latency_s if self.verification else 0.0)

    @property
    def cost_usd(self) -> float:
        return self.generation.cost_usd + (self.verification.cost_usd if self.verification else 0.0)

    @property
    def refused(self) -> bool:
        return is_refusal(self.final_answer)

    @property
    def poisoned_context(self) -> bool:
        return any(d.adversarial for d in self.context)


class DefensePipeline:
    """Compose retrieval (R), prompt assembly (F), generation (G) and verification (V).

    ``generator_models`` maps an alignment level to a model id so Layer 4 can be
    approximated with different checkpoints (e.g. a base vs. an instruct model)
    on the same provider.  Levels not in the map use the provider's default
    model together with that level's persona system prompt.
    """

    def __init__(
        self,
        corpus: DocCorpus,
        generator: ChatProvider,
        verifier: ChatProvider | None = None,
        *,
        policy: MetadataPolicy | None = None,
        top_k: int = 5,
        attack_strength: float = 1.0,
        generator_models: Mapping[str, str] | None = None,
        verifier_model: str | None = None,
        max_tokens: int = 512,
    ) -> None:
        self.corpus = corpus
        self.generator = generator
        self.verifier = verifier or generator
        self.policy = policy or MetadataPolicy()
        self.top_k = top_k
        self.attack_strength = attack_strength
        self.generator_models = dict(generator_models or {})
        self.verifier_model = verifier_model
        self.max_tokens = max_tokens

    # R ---------------------------------------------------------------------
    def retrieve(self, question: str, domain: str, rag: str) -> list[Document]:
        if rag == "none":
            return []
        if rag == "vanilla":
            return self.corpus.retrieve(question, domain, top_k=self.top_k, include_adversarial=False)
        if rag == "poisoned":
            return self.corpus.retrieve(question, domain, top_k=self.top_k, include_adversarial=True, attack_strength=self.attack_strength)
        return self.corpus.retrieve(
            question, domain, top_k=self.top_k, policy=self.policy, include_adversarial=True, attack_strength=self.attack_strength
        )

    # G ---------------------------------------------------------------------
    def generate(self, prompt: str, alignment: str) -> ChatResponse:
        persona = ALIGNMENT_PERSONAS[alignment]
        return self.generator.chat(
            [{"role": "system", "content": persona.system_prompt}, {"role": "user", "content": prompt}],
            model=self.generator_models.get(alignment),
            temperature=persona.temperature,
            max_tokens=self.max_tokens,
        )

    # V ---------------------------------------------------------------------
    def verify(self, question: str, draft: str, context: Sequence[Document]) -> tuple[str, bool, list[dict[str, str]], ChatResponse]:
        v = verify_claims(self.verifier, question, draft, context, model=self.verifier_model, max_tokens=self.max_tokens * 2)
        corrected = v.parse_ok and v.corrected_required and v.final_answer != draft
        final = v.final_answer if v.parse_ok else draft
        return final, corrected, [{"claim": c.text, "judgment": c.judgment} for c in v.claims], v.response

    # P ---------------------------------------------------------------------
    def answer(self, question: str, domain: str, config: LayerConfig) -> PipelineResult:
        context = self.retrieve(question, domain, config.rag)
        prompt = format_prompt(question, context, config.prompt)
        gen = self.generate(prompt, config.alignment)
        draft = gen.text
        final, corrected, claims, ver = draft, False, [], None
        if config.verifier:
            final, corrected, claims, ver = self.verify(question, draft, context)
        return PipelineResult(question, domain, config, context, prompt, draft, final, corrected, claims, gen, ver)
