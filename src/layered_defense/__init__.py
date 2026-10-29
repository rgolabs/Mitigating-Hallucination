"""Layered defense framework for mitigating hallucinations in LLM pipelines.

Import :class:`LayeredDefense` to wrap any LLM call in your agent with the
four layers; use ``layered-defense run`` to measure the effect on a benchmark.

Four composable layers:

* L1 -- LLM-as-a-Judge verification of a draft answer.
* L2 -- metadata-filtered retrieval-augmented generation (RAG).
* L3 -- system prompt, few-shot examples and negative constraints.
* L4 -- alignment level of the generator (base vs. aligned model/persona).

The :mod:`layered_defense.experiment` module runs a cumulative ablation over
these layers and :mod:`layered_defense.report` renders the results as LaTeX
tables.  Nothing in this package fabricates model output: if a provider call
fails the error is recorded, never replaced by simulated numbers.
"""

from importlib.metadata import PackageNotFoundError, version

from .corpus import ADVERSARIAL_DOCS, DocCorpus, Document, MetadataPolicy
from .evaluation import Judgement, judge_response, summarize_runs, wilson_interval
from .guard import Guard, GuardedAnswer, GuardEvent, GuardPolicy, LayeredDefense
from .llm import LLM, AnthropicLLM, CallableLLM, LangChainLLM, OpenAILLM, achat, resolve_llm
from .pipeline import DefensePipeline, LayerConfig, PipelineResult
from .prompts import ALIGNMENT_PERSONAS, format_prompt
from .providers import (
    ChatResponse,
    MockProvider,
    OpenAICompatibleProvider,
    Pricing,
    ProviderError,
    make_provider,
)
from .retrieval import CallableRetriever, LexicalRetriever, Retriever, as_documents
from .verifier import Claim, Verdict, averify_claims, verify_claims

try:
    __version__ = version("llm-layered-defense")
except PackageNotFoundError:  # pragma: no cover - source checkout without install
    __version__ = "0.0.0"

__all__ = [
    "ADVERSARIAL_DOCS",
    "ALIGNMENT_PERSONAS",
    "AnthropicLLM",
    "CallableLLM",
    "CallableRetriever",
    "Claim",
    "Guard",
    "GuardEvent",
    "GuardPolicy",
    "GuardedAnswer",
    "LLM",
    "LangChainLLM",
    "LayeredDefense",
    "LexicalRetriever",
    "OpenAILLM",
    "Retriever",
    "Verdict",
    "achat",
    "as_documents",
    "averify_claims",
    "resolve_llm",
    "verify_claims",
    "ChatResponse",
    "DefensePipeline",
    "DocCorpus",
    "Document",
    "Judgement",
    "LayerConfig",
    "MetadataPolicy",
    "MockProvider",
    "OpenAICompatibleProvider",
    "PipelineResult",
    "Pricing",
    "ProviderError",
    "format_prompt",
    "judge_response",
    "make_provider",
    "summarize_runs",
    "wilson_interval",
    "__version__",
]
