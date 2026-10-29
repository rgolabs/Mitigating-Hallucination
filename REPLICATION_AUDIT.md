# Audit of the original scripts against the paper (2026-10-07)

This note records what the previous `defense_pipeline.py` / `run_experiments.py` /
`generate_paper.py` actually did, compared with what `paper.tex` claimed, and
what was changed in the packaged version.

## Findings that invalidated the previous replication numbers

1. **Fabricated fallback.** Whenever an API call failed, `generate_answer`,
   `run_verifier` and `evaluate_response` fell back to a simulator whose
   hallucination probabilities were the paper's own Table 5 values (0.384,
   0.274, 0.185, 0.093) and whose verifier "corrected" 80 % of drafts. A run
   with an invalid key therefore produced numbers that confirmed the paper.
   The committed `results.json` (50.0 / 0.0 / 16.7 / 0.0 / 0.0 % with
   latencies 1.09 / 1.20 / 1.16 / 1.64 / 3.95 s) matches the simulator's
   Gaussian latency means and fixed costs, not a real run. Re-running the old
   script on 2026-10-07 hit HTTP 401 on all 72 calls and still printed a
   "33.3 % → 0.0 %" trend.
2. **Original-study costs overwritten.** The working copy of `paper.tex` and
   `generate_paper.py` changed the *original* Table 5 cost column from the PDF's
   2.0 / 2.1 / 3.4 / 3.4 / 17.2 $/1k to 0.24 / 0.37 / 0.60 / 0.42 / 1.22, which
   are exactly the July replication's costs. The comparison table therefore
   compared the replication to itself. Restored from the PDF.
3. **Claims not supported by the data in the paper.** The July run showed
   0.0 % hallucination in every row, yet the "Key Replication Observations"
   asserted progressive reductions. Row 1 scored helpfulness 1.0 because the
   prompt told the model to answer only from supplied documents while
   supplying none, so every answer was "I do not know".
4. **Sample size.** The comment said 15 questions; the code sampled 2 per
   domain (12). A 12-question row has a ±25-point confidence interval.

## Mismatches between the code and the paper's method

| Paper | Old code | Packaged version |
|---|---|---|
| 1,200 questions, two human annotators (κ = 0.84) | 12 questions, LLM judge | Configurable sample, LLM judge, interval reported; stated in the paper |
| 7B model fine-tuned with SFT + PPO | Same hosted model, different system prompt and temperature | Same proxy by default; `--base-model/--aligned-model` for real checkpoints; stated in the paper |
| Retrieve k = 20, filter, re-rank to k | Filter then whitespace-overlap top 5 | Tokenised overlap, 20 candidates, filter, top 5 |
| 5 % poisoned index measured (Table 2) | Poisoned retrieval computed and discarded | `--ablation extended` adds clean and poisoned vanilla-RAG rows |
| Recency "last 5 years" | Hard-coded `>= 2022` | Relative to the current or a given year |
| Replication "using gemini-3.5-flash" | `openrouter/auto-beta` (a router; model unknown per call) | Models recorded in results provenance and table caption |
| Costs from token usage | Undocumented constants 0.3052 / 0.9592 $/M | API-reported cost or explicit `--pricing MODEL=IN:OUT` |

Other defects fixed: absolute `/Users/rj/...` paths; `generate_paper.py`
overwrote hand edits to `paper.tex`; `evaluate_response` swallowed every
exception silently; the `random` module was unseeded; the `poisoned` config
was unreachable.

## Remaining caveats for the paper

* The abstract still states "We construct a 1,200-question benchmark" and
  Section 7 describes fine-tuning a 7B model; nothing in this repository
  reproduces those parts of the original study. The replication section now
  says so explicitly.
* The bundled corpus and QA set are a synthetic Kaggle dataset, not the
  benchmark described in the original study.
