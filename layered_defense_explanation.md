# Layered Defense Framework: how each layer is implemented

The four security layers of the paper map onto the `layered_defense` package as follows.

## L1 — LLM-as-a-Judge verifier (`layered_defense.pipeline.DefensePipeline.verify`)

After the generator produces a draft, a second model receives the question, the retrieved context and the draft, extracts atomic claims, labels each `SUPPORTED`, `CONTRADICTED` or `UNVERIFIABLE`, and returns a JSON object with a corrected `final_answer` when anything is unsupported. The prompt is `VERIFIER_PROMPT` in `pipeline.py`. Unparsable verdicts let the draft through unchanged and are kept for audit; they are never silently "corrected".

## L2 — metadata-filtered RAG (`layered_defense.corpus`)

`DocCorpus.retrieve()` draws 20 lexical candidates (token overlap after stop-word removal), applies a `MetadataPolicy` as a hard filter and keeps the top 5. The policy checks domain match, a recency window relative to the current year (default five years) and a trust/quality score (default ≥ 0.7). `inject_adversarial()` adds six adversarial documents (about 5 % of the store); in `rag="poisoned"` mode they take part in retrieval with an attacker-controlled similarity lift, in `rag="filtered"` mode the policy removes them because of their low trust score, stale date or wrong domain.

## L3 — prompt engineering and negative prompting (`layered_defense.prompts`)

`format_prompt()` assembles `σ ‖ examples ‖ context ‖ N ‖ q` for the five prompt conditions of the paper (`bare`, `system_only`, `system_3shot`, `system_negative`, `full`). Negative constraints are context-aware: with documents they forbid leaving the documents, closed-book they forbid fabrication but do not force a refusal.

## L4 — alignment (`layered_defense.prompts.ALIGNMENT_PERSONAS`)

The package cannot fine-tune weights. Each alignment level (`base`, `sft`, `aligned`) is a persona system prompt plus a sampling temperature, and `DefensePipeline(generator_models={...})` or `--base-model/--aligned-model` can point each level at a different checkpoint for a faithful comparison.

## Experiment and reporting

`layered_defense.experiment.run_ablation()` runs the cumulative rows of Table 5 (`--ablation table5`) or those rows plus the clean- and poisoned-index controls (`--ablation extended`), grades every final answer with an LLM judge (`layered_defense.evaluation`), records provider errors instead of inventing values, and summarises each row with its sample size, Wilson 95 % interval, refusal rate, correction rate, poisoned-context rate, latency, cost and helpfulness. `layered-defense report` renders LaTeX tables that `paper.tex` includes.
