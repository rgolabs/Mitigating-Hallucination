# Changelog

## 0.2.0 - 2026-10-07

Runtime guard for agents (phase one).

* `LayeredDefense` / `Guard`: wrap any LLM call with retrieval, prompt
  hardening, generation and claim-level verification; sync `answer()`,
  async `aanswer()`, `stream()` (tokens, then verdict and final answer) and
  standalone `verify()` for answers produced elsewhere.
* Bring your own LLM: `"provider:model"` strings, plain callables (sync or
  async), `OpenAILLM`, `AnthropicLLM`, `LangChainLLM`, or any `.chat()` object.
* `GuardPolicy`: rewrite / refuse / flag / allow per claim class, risk-tiered
  verification, parametric-knowledge mode, context budget, persona and prompt.
* Pluggable retrieval (`Retriever` protocol, `CallableRetriever`,
  `LexicalRetriever`) and `as_documents()` so tool outputs, dicts and
  LangChain documents can be passed as evidence.
* Adapters: tool definitions and functions for OpenAI/Anthropic tool calling,
  LangChain `as_runnable()`.
* `GuardEvent` callbacks for tracing; verifier claims carry evidence ids.

## 0.1.0 - 2026-10-07

First packaged release, replacing the three loose scripts.

* `layered_defense` package with corpus/retrieval, prompt assembly, providers,
  pipeline, LLM judge, ablation runner, reporter and `layered-defense` CLI.
* Removed the simulation fallback that fabricated results calibrated to the
  paper's own numbers whenever an API call failed; failures are now recorded.
* Closed-book prompts no longer force a refusal through the "stay inside the
  documents" constraint.
* Metadata recency window is relative to the current (or a given) year.
* Retrieval tokenises properly (punctuation, stop-words) and follows the
  paper's retrieve-20 / filter / re-rank-to-k scheme.
* Added the poisoned-index control conditions so Layer 2 is measured.
* Summary statistics carry `n`, error counts, Wilson 95 % intervals, refusal
  and correction rates; reporter emits data-driven observations.
* Bundled the synthetic corpus and QA data; no hard-coded absolute paths.
