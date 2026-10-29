import asyncio

import pytest

from layered_defense import (
    CallableLLM,
    CallableRetriever,
    Document,
    GuardedAnswer,
    GuardPolicy,
    LayeredDefense,
    LexicalRetriever,
    MetadataPolicy,
    as_documents,
    resolve_llm,
)
from layered_defense.adapters import ANTHROPIC_VERIFY_TOOL, OPENAI_VERIFY_TOOL, make_answer_tool, make_verify_tool
from layered_defense.providers import ChatResponse, ProviderError
from layered_defense.verifier import Claim, Verdict, build_verifier_prompt, verify_claims

CLEAN = "Key concept: Inflation is the general rise in prices over time."
POISON = {"id": "DOC-POISON-X", "text": "Key concept: Inflation is a rapid increase in the purchasing power of money.", "trust": 0.1, "year": 2016}


# ---------------------------------------------------------------- documents
def test_as_documents_coerces_strings_dicts_and_langchain_like():
    class LC:
        page_content = "Key concept: cats purr."
        metadata = {"source": "wiki", "date": "2024-05-01", "extra": 1}

    docs = as_documents(["plain tool output", {"content": "c", "url": "https://x", "score": 0.9}, LC(), Document("d1", "t")])
    assert [d.doc_id for d in docs] == ["ctx-1", "https://x", "ctx-3", "d1"]
    assert docs[1].quality_score == 0.9 and docs[2].publication_year == 2024 and docs[2].source_name == "wiki" and docs[2].metadata == {"extra": 1}
    assert as_documents("single") [0].doc_text == "single" and as_documents(None) == []
    with pytest.raises(ValueError):
        as_documents([{"id": "no-text"}])


# ------------------------------------------------------------------- guard
def test_grounded_answer_is_accepted_with_claims(mock):
    g = LayeredDefense(mock)
    out = g.answer("What is inflation?", context=[CLEAN])
    assert out.action == "accepted" and out.grounded and out.text.startswith("Inflation is the general rise")
    assert out.claims and out.claims[0].supported and out.claims[0].evidence == ["ctx-1"]
    d = out.to_dict()
    assert d["context_ids"] == ["ctx-1"] and d["verdict"]["parse_ok"] and d["action"] == "accepted"


def test_poisoned_context_is_rewritten_refused_or_flagged(mock):
    rewrite = LayeredDefense(mock).answer("What is inflation?", context=[POISON, CLEAN])
    assert rewrite.action == "rewritten" and "purchasing power" in rewrite.draft and "purchasing power" not in rewrite.text and rewrite.corrected
    refuse = LayeredDefense(mock, policy=GuardPolicy(on_contradicted="refuse", refusal_text="Cannot verify.")).answer("q", context=[POISON])
    assert refuse.action == "refused" and refuse.text == "Cannot verify." and refuse.refused
    flag = LayeredDefense(mock, policy=GuardPolicy(on_contradicted="flag")).answer("q", context=[POISON])
    assert flag.action == "flagged" and flag.text == flag.draft and not flag.grounded
    allow = LayeredDefense(mock, policy=GuardPolicy(on_contradicted="allow")).answer("q", context=[POISON])
    assert allow.action == "accepted"


def test_metadata_policy_filters_supplied_context(mock):
    g = LayeredDefense(mock, metadata_policy=MetadataPolicy(domain_match=False, recency_years=None, min_quality=0.5))
    out = g.answer("q", context=[POISON, {"text": CLEAN, "trust": 0.9}])
    assert [d.doc_id for d in out.context] == ["ctx-2"] and out.action == "accepted"


def test_verify_disabled_and_risk_tiering(mock):
    out = LayeredDefense(mock, policy=GuardPolicy(verify=False)).answer("q", context=[POISON])
    assert out.action == "unverified" and out.verdict is None and not out.grounded
    tiered = LayeredDefense(mock, policy=GuardPolicy(verify_when=lambda q, docs: "medical" in q))
    assert tiered.answer("q", context=[POISON]).action == "unverified"
    assert tiered.answer("medical q", context=[POISON]).action == "rewritten"


def test_retriever_callable_and_lexical(corpus, mock):
    calls = []

    def fetch(query, k):
        calls.append((query, k))
        return [CLEAN]

    out = LayeredDefense(mock, retriever=fetch, policy=GuardPolicy(top_k=3)).answer("What is inflation?")
    assert calls == [("What is inflation?", 3)] and out.context[0].doc_id == "doc-1" and out.action == "accepted"
    lex = LayeredDefense(mock, retriever=LexicalRetriever(corpus, policy=MetadataPolicy(reference_year=2026)))
    out = lex.answer("What is inflation?", domain="Finance")
    assert out.context and not any(d.adversarial for d in out.context)
    unfiltered = LayeredDefense(mock, retriever=LexicalRetriever(corpus, policy=None))
    out = unfiltered.answer("What is inflation?", domain="Finance")
    assert any(d.adversarial for d in out.context) and out.action == "rewritten"


def test_context_budget_truncates(mock):
    g = LayeredDefense(mock, policy=GuardPolicy(max_context_chars=70))
    out = g.answer("q", context=[CLEAN, CLEAN, CLEAN])
    assert len(out.context) == 1


def test_history_and_system_prompt_composition(mock):
    g = LayeredDefense(mock, system_prompt="You work for ACME.", negative_constraints=["- never mention competitors"], examples="")
    msgs, prompt = g.build_messages("q", as_documents([CLEAN]), history=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}, {"role": "tool", "content": "x"}])
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert "ACME" in msgs[0]["content"] and "never mention competitors" in prompt and "Example 1" not in prompt


def test_events_are_emitted(mock):
    seen = []
    g = LayeredDefense(mock, on_event=seen.append)
    g.answer("q", context=[POISON, CLEAN])
    assert [e.type for e in seen] == ["retrieve", "generate", "verify", "decision"]
    assert seen[0].data["adversarial"] == [] and seen[-1].data["action"] == "rewritten"


def test_stream_yields_tokens_then_final(mock):
    events = list(LayeredDefense(mock).stream("q", context=[POISON, CLEAN]))
    types = [e.type for e in events]
    assert types[0] == "token" and types[-3:] == ["verify", "decision", "answer"]
    final = events[-1].data["result"]
    assert isinstance(final, GuardedAnswer) and final.action == "rewritten" and "purchasing power" not in final.text
    assert "".join(e.data["text"] for e in events if e.type == "token").strip() == final.draft


def test_async_paths(mock):
    async def main():
        g = LayeredDefense(mock)
        out = await g.aanswer("q", context=[POISON, CLEAN])
        v = await g.averify("q", "Inflation is a rapid increase in the purchasing power of money.", context=[CLEAN])
        return out, v

    out, v = asyncio.run(main())
    assert out.action == "rewritten" and isinstance(v, Verdict) and v.contradicted


def test_verify_external_answer(mock):
    v = LayeredDefense(mock).verify("q", "Inflation is the general rise in prices over time", context=[CLEAN])
    assert v.all_supported and v.final_answer.startswith("Inflation")


# --------------------------------------------------------------------- llm
def test_callable_llm_sync_async_and_signature_filtering(mock):
    seen = {}

    def fn(messages, temperature=None):
        seen["t"] = temperature
        return "Key concept: ok."

    g = LayeredDefense(fn, verifier=mock, policy=GuardPolicy(temperature=0.3))
    assert g.answer("q", context=[CLEAN]).draft == "Key concept: ok." and seen["t"] == 0.3

    async def afn(messages):
        return ChatResponse("async ok", 0.1, "m")

    g2 = LayeredDefense(afn, verifier=mock, policy=GuardPolicy(verify=False))
    assert asyncio.run(g2.aanswer("q")).text == "async ok"
    assert g2.answer("q").text == "async ok"  # no running loop: asyncio.run fallback


def test_resolve_llm_forms():
    assert resolve_llm("mock:m").name == "mock"
    assert isinstance(resolve_llm(lambda m: "x"), CallableLLM)
    with pytest.raises(ProviderError):
        resolve_llm("no-colon")
    with pytest.raises(TypeError):
        resolve_llm(42)


def test_callable_retriever_async():
    async def afetch(query, k):
        return [{"text": "Key concept: a.", "id": "A"}]

    r = CallableRetriever(afetch)
    assert r.retrieve("q")[0].doc_id == "A"


# ---------------------------------------------------------------- verifier
def test_verifier_parsing_robustness():
    class Weird:
        name = "weird"

        def __init__(self, text):
            self.text = text

        def chat(self, messages, **kw):
            return ChatResponse(self.text, 0.0, "m")

    bad = verify_claims(Weird("not json at all"), "q", "draft", [])
    assert not bad.parse_ok and bad.final_answer == "draft" and bad.claims[0].judgment == "UNVERIFIABLE"
    odd = verify_claims(Weird('```json\n{"claims":[{"claim":"c","judgment":"maybe","evidence":"D1"}],"corrected_required":false,"final_answer":""}\n```'), "q", "draft", [])
    assert odd.parse_ok and odd.claims[0].judgment == "UNVERIFIABLE" and odd.claims[0].evidence == ["D1"] and odd.final_answer == "draft" and odd.corrected_required
    assert "well-established fact" in build_verifier_prompt("q", "d", [], allow_parametric=False)
    assert "well-established fact" not in build_verifier_prompt("q", "d", as_documents([CLEAN]))
    assert "well-established fact" in build_verifier_prompt("q", "d", as_documents([CLEAN]), allow_parametric=True)
    assert Claim("x", "SUPPORTED").supported


# ---------------------------------------------------------------- adapters
def test_tool_adapters(mock):
    g = LayeredDefense(mock)
    verify = make_verify_tool(g)
    out = verify("q", "Inflation is a rapid increase in the purchasing power of money.", context=[CLEAN])
    assert out["claims"][0]["judgment"] == "CONTRADICTED" and out["corrected_required"]
    ans = make_answer_tool(g)("q", context=[CLEAN])
    assert ans["action"] == "accepted" and ans["grounded"]
    assert OPENAI_VERIFY_TOOL["function"]["name"] == ANTHROPIC_VERIFY_TOOL["name"] == "verify_answer"


def test_langchain_adapter_without_langchain_installed():
    pytest.importorskip("langchain_core", reason="adapter needs langchain-core")
