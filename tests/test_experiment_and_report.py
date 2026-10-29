import json

import pytest

from layered_defense import DefensePipeline, MockProvider, ProviderError, summarize_runs, wilson_interval
from layered_defense.cli import DATA, main
from layered_defense.experiment import ABLATION_EXTENDED, ABLATION_TABLE5, load_questions, run_ablation, sample_questions
from layered_defense.report import ORIGINAL_TABLE5, observations, render_latex, render_markdown


def test_wilson_interval():
    lo, hi = wilson_interval(0, 12)
    assert lo == 0.0 and 0.2 < hi < 0.3
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_sampling_is_stratified_and_deterministic():
    qs = load_questions(DATA / "rag_qa_samples.csv")
    a, b = sample_questions(qs, 2, 7), sample_questions(qs, 2, 7)
    assert [q.sample_id for q in a] == [q.sample_id for q in b] and len(a) == 12
    assert sorted({q.domain for q in a}) == sorted({q.domain for q in qs})


def test_errors_are_recorded_not_simulated(corpus):
    class Flaky(MockProvider):
        def chat(self, messages, **kw):
            if self.calls == 0:
                self.calls += 1
                raise ProviderError("boom")
            return super().chat(messages, **kw)

    flaky = Flaky()
    qs = sample_questions(load_questions(DATA / "rag_qa_samples.csv"), 1, 1)[:2]
    res = run_ablation(DefensePipeline(corpus, flaky), flaky, qs, ABLATION_TABLE5[:1])
    assert res["raw_runs"][0]["error"].startswith("ProviderError")
    assert res["summary"][0]["errors"] == 1 and res["summary"][0]["n"] == 1


def test_summary_excludes_errors_from_rates():
    runs = [
        {"config_row": 0, "hallucinated": True, "refused": False, "corrected": False, "poisoned_context": False, "latency_s": 1.0, "cost_usd": 0.001, "cost_source": "api", "helpfulness": 2, "error": None},
        {"config_row": 0, "hallucinated": False, "refused": True, "corrected": False, "poisoned_context": False, "latency_s": 3.0, "cost_usd": 0.0, "cost_source": "none", "helpfulness": 3, "error": None},
        {"config_row": 0, "error": "ProviderError: x"},
    ]
    s = summarize_runs(runs, [{"row": 0, "name": "r0"}])[0]
    assert s["n"] == 2 and s["errors"] == 1 and s["hr_pct"] == 50.0 and s["refusal_pct"] == 50.0
    assert s["latency_s"] == 2.0 and s["cost_per_1k"] == 1.0 and s["helpful"] == 2.5


def test_original_table_matches_pdf():
    assert ORIGINAL_TABLE5[4] == {"hr": 6.1, "lat": 3.9, "cost": 17.2, "help": 4.3}


def test_cli_demo_and_report_roundtrip(tmp_path):
    out = tmp_path / "r.json"
    assert main(["demo", "--n-per-domain", "1", "--ablation", "extended", "--out", str(out)]) == 0
    data = json.loads(out.read_text())
    assert data["provenance"]["provider"] == "mock" and len(data["raw_runs"]) == 6 * len(ABLATION_EXTENDED)
    assert main(["report", str(out), "--format", "markdown"]) == 0
    assert main(["report", str(out)]) == 2  # mock results refused for LaTeX by default
    tex = tmp_path / "t.tex"
    assert main(["report", str(out), "--allow-mock", "--out", str(tex)]) == 0
    text = tex.read_text()
    assert "tab:replicated_results" in text and "17.2" in text and "\\item" in text


def test_observations_are_data_driven():
    base = {"configuration": "c", "n": 10, "errors": 0, "hr_ci95_pct": [0, 30], "refusal_pct": 0, "corrected_pct": None, "poisoned_context_pct": 0, "latency_s": 1.0, "cost_per_1k": None, "helpful": 4.0}
    results = {"provenance": {"domains": ["a"], "configs": []}, "summary": [{**base, "row": 0, "hr_pct": 10.0}, {**base, "row": 1, "hr_pct": 20.0}]}
    assert any("increases" in o for o in observations(results))
    results["summary"][1]["hr_pct"] = 5.0
    assert any("monotonically" in o for o in observations(results))
    assert "Mock" not in render_markdown(results)
    with pytest.raises(ValueError):
        render_latex({"provenance": {"provider": "mock"}, "summary": [], "raw_runs": []})


def test_normalize_base_url():
    from layered_defense.cli import _normalize_base_url

    assert _normalize_base_url("http://127.0.0.1:8787") == "http://127.0.0.1:8787/v1"
    assert _normalize_base_url("http://127.0.0.1:8787/") == "http://127.0.0.1:8787/v1"
    assert _normalize_base_url("http://h/v1/") == "http://h/v1"
    assert _normalize_base_url("https://x/openai/v1") == "https://x/openai/v1"


def test_workers_preserve_order(corpus):
    mock = MockProvider()
    qs = sample_questions(load_questions(DATA / "rag_qa_samples.csv"), 1, 3)
    seq = run_ablation(DefensePipeline(corpus, mock), mock, qs, ABLATION_TABLE5[:3])
    par = run_ablation(DefensePipeline(corpus, mock), mock, qs, ABLATION_TABLE5[:3], workers=4)
    key = lambda r: (r["sample_id"], r["config_row"], r["final_text"])  # noqa: E731
    assert [key(r) for r in seq["raw_runs"]] == [key(r) for r in par["raw_runs"]]
    assert par["summary"] == seq["summary"]
