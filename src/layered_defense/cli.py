"""Command-line interface: ``layered-defense run|report|demo``."""

from __future__ import annotations

import argparse
import json
import os
import sys
from importlib.resources import files
from pathlib import Path
from typing import Any

from . import __version__
from .corpus import DocCorpus, MetadataPolicy
from .experiment import ABLATIONS, load_questions, run_ablation, sample_questions
from .pipeline import DefensePipeline
from .providers import API_KEY_ENV, KNOWN_BASE_URLS, Pricing, ProviderError, make_provider
from .report import load_results, render_latex, render_markdown

DATA = files("layered_defense") / "data"


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover
        return
    load_dotenv(override=False)


def _parse_pricing(items: list[str] | None) -> dict[str, Pricing]:
    out: dict[str, Pricing] = {}
    for item in items or []:
        try:
            model, rest = item.split("=", 1)
            inp, outp = rest.split(":", 1)
            out[model] = Pricing(float(inp), float(outp))
        except ValueError as err:
            raise SystemExit(f"--pricing expects MODEL=INPUT_USD_PER_M:OUTPUT_USD_PER_M, got {item!r}") from err
    return out


def _normalize_base_url(url: str) -> str:
    """Append ``/v1`` when the URL has no path: OpenAI-style servers mount there."""
    from urllib.parse import urlparse

    path = urlparse(url).path.rstrip("/")
    return url.rstrip("/") + ("/v1" if not path else "")


def _provider(args: argparse.Namespace, model: str, pricing: dict[str, Pricing]):
    provider = args.provider
    base_url, api_key = args.base_url, args.api_key
    if provider == "local":
        base_url = base_url or os.environ.get("LOCAL_LLM_BASE_URL", "http://127.0.0.1:8787/v1")
        api_key = api_key or os.environ.get("LOCAL_LLM_API_KEY")
        provider = "custom"
    if provider == "custom" and not base_url:
        raise SystemExit("--provider custom requires --base-url")
    if base_url:
        base_url = _normalize_base_url(base_url)
    if provider == "custom":
        api_key = api_key or os.environ.get("LLM_API_KEY")
    return make_provider(provider, model=model, api_key=api_key, base_url=base_url, pricing=pricing.get(model), timeout=args.timeout)


def cmd_run(args: argparse.Namespace) -> int:
    _load_dotenv()
    pricing = _parse_pricing(args.pricing)
    gen_model = args.model
    verifier_model = args.verifier_model or gen_model
    judge_model = args.judge_model or verifier_model
    try:
        generator = _provider(args, gen_model, pricing)
        verifier = _provider(args, verifier_model, pricing)
        judge = _provider(args, judge_model, pricing)
    except ProviderError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2

    corpus = DocCorpus.from_csv(args.corpus or DATA / "documents_corpus.csv").inject_adversarial()
    questions = sample_questions(load_questions(args.questions or DATA / "rag_qa_samples.csv"), args.n_per_domain, args.seed)
    policy = MetadataPolicy(min_quality=args.min_quality, recency_years=args.recency_years, reference_year=args.reference_year)
    generator_models = {k: v for k, v in (("base", args.base_model), ("aligned", args.aligned_model)) if v}
    pipeline = DefensePipeline(
        corpus, generator, verifier, policy=policy, top_k=args.top_k, attack_strength=args.attack_strength,
        generator_models=generator_models, verifier_model=verifier_model,
    )
    configs = ABLATIONS[args.ablation]
    total = len(questions) * len(configs)
    print(f"layered-defense {__version__}: {len(questions)} questions x {len(configs)} configs = {total} runs via {generator.name}", file=sys.stderr)

    done = {"n": 0}

    def progress(rec: dict[str, Any]) -> None:
        done["n"] += 1
        status = f"ERROR {rec['error']}" if rec.get("error") else f"HR={'Y' if rec['hallucinated'] else 'n'} help={rec['helpfulness']} lat={rec['latency_s']:.1f}s"
        print(f"[{done['n']:>3}/{total}] row {rec['config_row']} {rec['sample_id']} [{rec['domain']}] {status}", file=sys.stderr)

    provenance = {
        "package_version": __version__,
        "provider": generator.name,
        "base_url": getattr(generator, "base_url", None),
        "models": {"generator": gen_model, "generator_by_alignment": generator_models, "verifier": verifier_model, "judge": judge_model},
        "seed": args.seed,
        "n_per_domain": args.n_per_domain,
        "ablation": args.ablation,
        "workers": args.workers,
        "top_k": args.top_k,
        "attack_strength": args.attack_strength,
        "policy": {"min_quality": policy.min_quality, "recency_years": policy.recency_years, "min_year": policy.min_year()},
        "pricing": {k: {"input_per_m": v.input_per_m, "output_per_m": v.output_per_m} for k, v in pricing.items()},
        "corpus_docs": len(corpus.docs),
        "adversarial_docs": len(corpus.adversarial_docs),
    }
    results = run_ablation(pipeline, judge, questions, configs, judge_model=judge_model, sleep_s=args.sleep, provenance=provenance, on_result=progress, workers=args.workers)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(render_markdown(results))
    print(f"results written to {out}", file=sys.stderr)
    return 1 if all(s["n"] == 0 for s in results["summary"]) else 0


def cmd_report(args: argparse.Namespace) -> int:
    results = load_results(args.results)
    if args.format == "markdown":
        text = render_markdown(results)
    else:
        try:
            text = render_latex(results, allow_mock=args.allow_mock)
        except ValueError as err:
            print(f"error: {err}", file=sys.stderr)
            return 2
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"report written to {args.out}", file=sys.stderr)
    else:
        print(text)
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    ns = argparse.Namespace(**{**vars(args), "provider": "mock", "model": "mock-model", "base_url": None, "api_key": None})
    return cmd_run(ns)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="layered-defense", description=__doc__)
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    def add_run_options(sp: argparse.ArgumentParser, with_provider: bool) -> None:
        if with_provider:
            sp.add_argument("--provider", default="openrouter", choices=[*KNOWN_BASE_URLS, "local", "custom", "mock"],
                            help="chat provider; keys are read from " + ", ".join(f"{k}: ${v}" for k, v in API_KEY_ENV.items()) + ", local: $LOCAL_LLM_API_KEY, custom: $LLM_API_KEY")
            sp.add_argument("--base-url", help="OpenAI-compatible base URL (for --provider custom/local)")
            sp.add_argument("--api-key", help="API key (prefer the environment variable)")
            sp.add_argument("--model", required=True, help="generator model id")
        sp.add_argument("--base-model", help="model id to use for the 'base' alignment level (e.g. a non-instruct checkpoint)")
        sp.add_argument("--aligned-model", help="model id to use for the 'aligned' alignment level")
        sp.add_argument("--verifier-model", help="model id for the L1 verifier (default: generator model)")
        sp.add_argument("--judge-model", help="model id for the evaluation judge (default: verifier model)")
        sp.add_argument("--pricing", action="append", metavar="MODEL=IN:OUT", help="USD per 1M input:output tokens, used when the API does not report cost; repeatable")
        sp.add_argument("--ablation", default="table5", choices=sorted(ABLATIONS))
        sp.add_argument("--n-per-domain", type=int, default=5, help="questions sampled per domain (default 5 => 30 questions)")
        sp.add_argument("--seed", type=int, default=42)
        sp.add_argument("--corpus", help="documents CSV (default: bundled synthetic corpus)")
        sp.add_argument("--questions", help="QA CSV (default: bundled synthetic QA set)")
        sp.add_argument("--top-k", type=int, default=5)
        sp.add_argument("--attack-strength", type=float, default=1.0, help="0..1 how retrieval-optimised the adversarial docs are")
        sp.add_argument("--min-quality", type=float, default=0.7)
        sp.add_argument("--recency-years", type=int, default=5)
        sp.add_argument("--reference-year", type=int, help="year the recency window is measured from (default: today)")
        sp.add_argument("--sleep", type=float, default=0.0, help="seconds to sleep between runs (rate limiting)")
        sp.add_argument("--workers", type=int, default=1, help="concurrent runs (keep at 1 for strict rate limits)")
        sp.add_argument("--timeout", type=float, default=120.0)
        sp.add_argument("--out", default="results.json")

    run = sub.add_parser("run", help="run the cumulative ablation against a live provider")
    add_run_options(run, with_provider=True)
    run.set_defaults(func=cmd_run)

    demo = sub.add_parser("demo", help="run the ablation offline with the deterministic mock provider")
    add_run_options(demo, with_provider=False)
    demo.set_defaults(func=cmd_demo, out="results_mock.json")

    rep = sub.add_parser("report", help="render a results file as LaTeX tables or Markdown")
    rep.add_argument("results")
    rep.add_argument("--format", default="latex", choices=["latex", "markdown"])
    rep.add_argument("--out")
    rep.add_argument("--allow-mock", action="store_true", help="render mock-provider results (never for a paper)")
    rep.set_defaults(func=cmd_report)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
