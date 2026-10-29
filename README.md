# llm-layered-defense

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23768596.svg)](https://doi.org/10.5281/zenodo.23768596)

A small, dependency-light framework and experiment harness for the paper
*"A Layered Defense Framework for Mitigating Hallucinations in Large Language
Models"* (Govindarajan, 2026). It implements the four security layers as
composable pieces and runs the cumulative ablation of Table 5 against any
OpenAI-compatible chat endpoint.

| Layer | Module | What it does |
|---|---|---|
| L1 verifier | `layered_defense.pipeline.DefensePipeline.verify` | LLM-as-a-Judge extracts atomic claims, labels them supported / contradicted / unverifiable and rewrites the draft. |
| L2 metadata-filtered RAG | `layered_defense.corpus` | Lexical top-k retrieval over a document store; a `MetadataPolicy` (domain, recency, trust score) is applied as a hard filter before re-ranking. A poisoned index with 5 % adversarial documents is supported for the context-poisoning experiment. |
| L3 prompt engineering | `layered_defense.prompts` | System instruction, 3-shot grounded examples and negative constraints (context-aware: closed-book prompts get anti-fabrication constraints instead of a forced refusal). |
| L4 alignment | `layered_defense.prompts.ALIGNMENT_PERSONAS` + `--base-model/--aligned-model` | Approximated on hosted models by persona + temperature, or faithfully by pointing each level at a different checkpoint. |

The harness **never simulates model output**. A failed provider call is
recorded on the run with an `error` field and excluded from every statistic.
An offline `mock` provider exists for tests and dry runs; its results are
tagged and refused by the LaTeX reporter unless `--allow-mock` is passed.

## Use it inside your agent

```python
from layered_defense import LayeredDefense, GuardPolicy

# any LLM: "openrouter:google/gemini-2.5-flash", "openai:gpt-4o", "custom:model"
# (with base_url), a callable fn(messages) -> str, OpenAILLM(client, model),
# AnthropicLLM(client, model) or LangChainLLM(chat_model)
guard = LayeredDefense("openrouter:google/gemini-2.5-flash",
                       verifier="openrouter:anthropic/claude-sonnet-4.5",
                       system_prompt="You are the support assistant for ACME.")

out = guard.answer("Is this customer eligible for a refund?",
                   context=[kb_passage, crm_ticket])   # tool outputs as evidence
out.text      # grounded answer (rewritten / refused / flagged per policy)
out.action    # accepted | rewritten | refused | flagged | unverified
out.claims    # [Claim(text, judgment, evidence_ids), ...]

await guard.aanswer(...)                 # async
for ev in guard.stream(...): ...         # token events, then verify/decision/answer
guard.verify(question, answer, context)  # audit an answer produced elsewhere
```

`GuardPolicy(on_contradicted="refuse", on_unverifiable="flag",
allow_parametric=True, verify_when=lambda q, docs: is_high_risk(q))` controls
what happens per claim class and when the verifier runs. Pass a
`retriever=` (any `retrieve(query, k=, domain=)` object or `fn(query, k)`),
`metadata_policy=` for Layer 2 filtering, and `on_event=` for tracing.
Tool-calling adapters live in `layered_defense.adapters`
(`OPENAI_VERIFY_TOOL`, `ANTHROPIC_VERIFY_TOOL`, `make_verify_tool(guard)`,
`as_runnable(guard)` for LangChain/LangGraph).

## Install

```bash
pip install llm-layered-defense
```

Requires Python 3.10+. A bundled synthetic corpus (120 documents) and QA set
(500 questions, six domains) ship with the package, so no download is needed.

## Quick start

```bash
# offline dry run with the deterministic mock provider
layered-defense demo --n-per-domain 2 --ablation extended

# live run through OpenRouter
export OPENROUTER_API_KEY=...
layered-defense run --provider openrouter --model google/gemini-2.5-flash \
    --verifier-model anthropic/claude-sonnet-4.5 --n-per-domain 5 --out results.json

# any OpenAI-compatible server (vLLM, Ollama, LM Studio, a local proxy, ...)
layered-defense run --provider custom --base-url http://127.0.0.1:8787/v1 \
    --api-key "$LLM_API_KEY" --model my-model --pricing my-model=1:5

# render LaTeX tables (+ data-driven observations) or a Markdown summary
layered-defense report results.json --out results_tables.tex
layered-defense report results.json --format markdown
```

Providers: `openrouter`, `openai`, `gemini` (OpenAI-compatible endpoint),
`groq`, `ollama`, `local` (`$LOCAL_LLM_BASE_URL` / `$LOCAL_LLM_API_KEY`),
`custom` (`--base-url`), `mock`. A `.env` file in the working directory is
loaded automatically.

### Ablations

`--ablation table5` runs the five cumulative rows of the paper. `--ablation
extended` adds two Layer-2 controls between rows 1 and 2, *vanilla RAG on a
clean index* and *vanilla RAG on the poisoned index*, so the context-poisoning
effect that motivates metadata filtering is measured rather than assumed.
`--attack-strength` (0..1) sets how retrieval-optimised the adversarial
documents are; at 1 they tie with the best clean document.

### What a results file contains

`provenance` (models, seed, policy, pricing, timestamps), `summary` (per row:
`n`, `errors`, hallucination rate with a 95 % Wilson interval, refusal rate,
verifier-correction rate, share of prompts that contained an adversarial
document, mean latency, cost per 1k queries, mean helpfulness) and `raw_runs`
(every prompt's context ids, draft, final answer, verifier claims, judge
verdict and rationale).

## Library use

```python
from layered_defense import DefensePipeline, DocCorpus, LayerConfig, MetadataPolicy, make_provider

corpus = DocCorpus.from_csv("documents_corpus.csv").inject_adversarial()
llm = make_provider("openrouter", model="google/gemini-2.5-flash")
pipe = DefensePipeline(corpus, llm, policy=MetadataPolicy(min_quality=0.7, recency_years=5))
full_stack = LayerConfig("full", rag="filtered", prompt="full", alignment="aligned", verifier=True)
result = pipe.answer("What is inflation?", "Finance", full_stack)
print(result.final_answer, result.corrected, result.latency_s, result.cost_usd)
```

## Honest scope

* Layer 4 cannot fine-tune anything. With a single hosted model the "base"
  and "aligned" rows differ only by system prompt and temperature. Use
  `--base-model` / `--aligned-model` with real base vs. instruct checkpoints
  for a faithful L4 comparison.
* Hallucination is judged by an LLM, not by human annotators. Use a judge at
  least as strong as the generator and a different model where possible.
* Retrieval is lexical (token overlap with stop-word removal), not embedding
  based. Swap `DocCorpus.retrieve` for your own retriever if needed.
* Small samples give wide intervals. The reporter prints the Wilson interval
  and refuses to claim a trend the data do not show.

## Paper

Preprint: Govindarajan, R. (2026). *A Layered Defense Framework for Mitigating Hallucinations in Large Language Models.* Zenodo. https://doi.org/10.5281/zenodo.23768596

`paper.tex` includes `results_tables.tex`, which `layered-defense report`
generates from `results.json`. Build with `latexmk -pdf paper.tex`.

## Development

```bash
uv venv && uv pip install -e ".[dev]"
pytest
ruff check src tests
python -m build && twine check dist/*
```

## License

MIT. The bundled corpus and QA set are synthetic (CC0-style educational use).
