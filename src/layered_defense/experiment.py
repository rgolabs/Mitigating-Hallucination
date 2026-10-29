"""Cumulative ablation experiment over the four defense layers."""

from __future__ import annotations

import csv
import datetime as _dt
import random
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .evaluation import judge_response, summarize_runs
from .pipeline import DefensePipeline, LayerConfig
from .providers import ChatProvider, ProviderError

# Rows of Table 5 in the paper, turned on cumulatively.
ABLATION_TABLE5: tuple[LayerConfig, ...] = (
    LayerConfig("Base model, no defenses", rag="none", prompt="bare", alignment="base", row=0),
    LayerConfig("+ L3 prompt engineering", rag="none", prompt="full", alignment="base", row=1),
    LayerConfig("+ L2 metadata-filtered RAG", rag="filtered", prompt="full", alignment="base", row=2),
    LayerConfig("+ L4 aligned generator", rag="filtered", prompt="full", alignment="aligned", row=3),
    LayerConfig("+ L1 LLM-as-a-Judge verifier (full stack)", rag="filtered", prompt="full", alignment="aligned", verifier=True, row=4),
)

# Table 5 plus the two Layer-2 control conditions from Table 2 (clean and
# poisoned vanilla RAG) so the context-poisoning effect is actually measured.
ABLATION_EXTENDED: tuple[LayerConfig, ...] = (
    ABLATION_TABLE5[0],
    ABLATION_TABLE5[1],
    LayerConfig("+ vanilla RAG, clean index (control)", rag="vanilla", prompt="full", alignment="base", row=11),
    LayerConfig("+ vanilla RAG, poisoned index (control)", rag="poisoned", prompt="full", alignment="base", row=12),
    ABLATION_TABLE5[2],
    ABLATION_TABLE5[3],
    ABLATION_TABLE5[4],
)
ABLATIONS = {"table5": ABLATION_TABLE5, "extended": ABLATION_EXTENDED}


@dataclass(frozen=True)
class Question:
    sample_id: str
    domain: str
    question: str
    expected_answer: str

    @classmethod
    def from_record(cls, rec: dict[str, Any]) -> Question:
        return cls(str(rec["sample_id"]), str(rec["domain"]), str(rec["question"]), str(rec["expected_answer"]))


def load_questions(path: str | Path) -> list[Question]:
    with open(path, newline="", encoding="utf-8") as fh:
        return [Question.from_record(r) for r in csv.DictReader(fh)]


def sample_questions(questions: Sequence[Question], n_per_domain: int, seed: int = 42) -> list[Question]:
    """Stratified sample of ``n_per_domain`` questions per domain, shuffled."""
    rng = random.Random(seed)
    by_domain: dict[str, list[Question]] = defaultdict(list)
    for q in questions:
        by_domain[q.domain].append(q)
    chosen: list[Question] = []
    for domain in sorted(by_domain):
        pool = sorted(by_domain[domain], key=lambda q: q.sample_id)
        chosen.extend(rng.sample(pool, min(n_per_domain, len(pool))))
    rng.shuffle(chosen)
    return chosen


def run_ablation(
    pipeline: DefensePipeline,
    judge: ChatProvider,
    questions: Sequence[Question],
    configs: Sequence[LayerConfig] = ABLATION_TABLE5,
    *,
    judge_model: str | None = None,
    sleep_s: float = 0.0,
    provenance: dict[str, Any] | None = None,
    on_result: Callable[[dict[str, Any]], None] | None = None,
    workers: int = 1,
) -> dict[str, Any]:
    """Run every configuration on every question and return results + summary.

    Provider failures are recorded on the run record under ``error`` and the
    run is excluded from the summary statistics.  No value is ever simulated.
    ``workers`` > 1 runs (question, config) pairs concurrently; the returned
    ``raw_runs`` keep question-major order regardless.
    """
    started = _dt.datetime.now(_dt.timezone.utc)
    lock = threading.Lock()

    def one(q: Question, cfg: LayerConfig) -> dict[str, Any]:
        rec: dict[str, Any] = {
            "sample_id": q.sample_id,
            "domain": q.domain,
            "question": q.question,
            "expected_answer": q.expected_answer,
            "config_row": cfg.row,
            "config_name": cfg.name,
            "config": cfg.as_dict(),
        }
        try:
            res = pipeline.answer(q.question, q.domain, cfg)
            verdict = judge_response(judge, q.question, res.final_answer, q.expected_answer, res.context, model=judge_model)
            rec.update(
                {
                    "context_ids": [d.doc_id for d in res.context],
                    "poisoned_context": res.poisoned_context,
                    "generated_text": res.draft_answer,
                    "final_text": res.final_answer,
                    "corrected": res.corrected,
                    "verifier_claims": res.verifier_claims,
                    "refused": res.refused,
                    "helpfulness": verdict.helpfulness,
                    "hallucinated": verdict.hallucinated,
                    "judge_rationale": verdict.rationale,
                    "latency_s": round(res.latency_s, 4),
                    "cost_usd": res.cost_usd,
                    "cost_source": res.generation.cost_source,
                    "generator_model": res.generation.model,
                    "prompt_tokens": res.generation.prompt_tokens + (res.verification.prompt_tokens if res.verification else 0),
                    "completion_tokens": res.generation.completion_tokens + (res.verification.completion_tokens if res.verification else 0),
                    "error": None,
                }
            )
        except (ProviderError, ValueError) as err:
            rec.update({"error": f"{type(err).__name__}: {err}"})
        if on_result:
            with lock:
                on_result(rec)
        if sleep_s:
            time.sleep(sleep_s)
        return rec

    jobs = [(q, cfg) for q in questions for cfg in configs]
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            runs = list(pool.map(lambda jc: one(*jc), jobs))
    else:
        runs = [one(q, cfg) for q, cfg in jobs]

    cfg_dicts = [c.as_dict() for c in configs]
    finished = _dt.datetime.now(_dt.timezone.utc)
    return {
        "provenance": {
            "started_at": started.isoformat(),
            "finished_at": finished.isoformat(),
            "n_questions": len(questions),
            "domains": sorted({q.domain for q in questions}),
            "sample_ids": [q.sample_id for q in questions],
            "configs": cfg_dicts,
            **(provenance or {}),
        },
        "summary": summarize_runs(runs, cfg_dicts),
        "raw_runs": runs,
    }
