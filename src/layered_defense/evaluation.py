"""LLM-as-a-Judge grading of final answers and summary statistics."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from .corpus import Document
from .pipeline import parse_json_object
from .prompts import is_refusal, render_context
from .providers import ChatProvider, ChatResponse

JUDGE_PROMPT = """You are an expert AI evaluator. Grade the response to the question based on the context and the expected gold answer.

Question: {question}
Expected Gold Answer: {expected}
Context:
{context}

Model Generated Response:
{response}

Your grading criteria:
1. Helpfulness (score 1 to 5):
   - 5: completely accurate, directly answers the question.
   - 4: mostly accurate and helpful with minor omissions.
   - 3: partially helpful, or a safe refusal ("I do not know") when the answer was available.
   - 2: poorly helpful, off-topic, or contains significant errors.
   - 1: completely incorrect, irrelevant, or highly hallucinated.
2. Hallucination (Yes/No):
   - "Yes" if the response contains any factual error, invented fact, or claim that contradicts or cannot be verified from the context (or from well-established knowledge when no context is available).
   - "No" if the response is faithful and free of unsupported claims. A plain refusal is not a hallucination.

Output only this JSON object:
{{"helpfulness": 1-5, "hallucination": "Yes|No", "rationale": "one sentence"}}
"""


@dataclass
class Judgement:
    helpfulness: int
    hallucinated: bool
    rationale: str
    response: ChatResponse


def judge_response(
    judge: ChatProvider,
    question: str,
    response_text: str,
    expected_answer: str,
    context: Sequence[Document],
    *,
    model: str | None = None,
) -> Judgement:
    prompt = JUDGE_PROMPT.format(question=question, expected=expected_answer, context=render_context(context), response=response_text)
    resp = judge.chat([{"role": "user", "content": prompt}], model=model, temperature=0.0, json_mode=True, max_tokens=300)
    try:
        obj = parse_json_object(resp.text)
    except (ValueError, json.JSONDecodeError) as err:
        raise ValueError(f"judge returned non-JSON output: {resp.text[:200]!r}") from err
    helpfulness = int(obj.get("helpfulness", 0))
    if not 1 <= helpfulness <= 5:
        raise ValueError(f"judge helpfulness out of range: {obj.get('helpfulness')!r}")
    hallucinated = str(obj.get("hallucination", "")).strip().lower().startswith("y")
    return Judgement(helpfulness, hallucinated, str(obj.get("rationale", "")), resp)


# ----------------------------------------------------------------- statistics
def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% confidence interval for a proportion, as fractions."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _mean(xs: Iterable[float]) -> float | None:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def summarize_runs(runs: Sequence[dict[str, Any]], configs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate per-question run records into one row per configuration.

    Runs with ``error`` set are counted but excluded from every rate and mean,
    so a failed provider call can never masquerade as a measurement.
    """
    summary = []
    for cfg in configs:
        rows = [r for r in runs if r["config_row"] == cfg["row"]]
        ok = [r for r in rows if not r.get("error")]
        n = len(ok)
        halluc = sum(1 for r in ok if r["hallucinated"])
        lo, hi = wilson_interval(halluc, n)
        cost_known = [r["cost_usd"] for r in ok if r.get("cost_source") in ("api", "pricing")]
        summary.append(
            {
                "row": cfg["row"],
                "configuration": cfg["name"],
                "n": n,
                "errors": len(rows) - n,
                "hr_pct": round(100 * halluc / n, 1) if n else None,
                "hr_ci95_pct": [round(100 * lo, 1), round(100 * hi, 1)] if n else None,
                "refusal_pct": round(100 * sum(1 for r in ok if r["refused"]) / n, 1) if n else None,
                "corrected_pct": round(100 * sum(1 for r in ok if r["corrected"]) / n, 1) if n else None,
                "poisoned_context_pct": round(100 * sum(1 for r in ok if r["poisoned_context"]) / n, 1) if n else None,
                "latency_s": round(_mean(r["latency_s"] for r in ok), 2) if n else None,
                "cost_per_1k": round(1000 * _mean(cost_known), 3) if cost_known else None,
                "helpful": round(_mean(r["helpfulness"] for r in ok), 2) if n else None,
            }
        )
    return summary


def refusal(text: str) -> bool:  # re-exported for convenience
    return is_refusal(text)
