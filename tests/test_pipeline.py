import pytest

from layered_defense import DefensePipeline, LayerConfig, MetadataPolicy, judge_response
from layered_defense.pipeline import parse_json_object


def test_layer_config_validation():
    with pytest.raises(ValueError):
        LayerConfig("x", rag="magic")
    with pytest.raises(ValueError):
        LayerConfig("x", alignment="dpo")
    assert LayerConfig("x", alignment="ppo").alignment == "ppo"  # alias kept


def test_parse_json_object_tolerates_fences_and_prose():
    assert parse_json_object('Sure:\n```json\n{"a": 1}\n```')["a"] == 1
    assert parse_json_object('text {"a": 2} trailing')["a"] == 2
    with pytest.raises(ValueError):
        parse_json_object("[1,2]")


def test_closed_book_bare_goes_through_without_context(corpus, mock):
    pipe = DefensePipeline(corpus, mock)
    res = pipe.answer("What is inflation?", "Finance", LayerConfig("r0"))
    assert res.context == [] and res.verification is None and not res.corrected
    assert res.latency_s == res.generation.latency_s


def test_poisoned_rag_copies_lie_and_verifier_corrects_it(corpus, mock):
    pipe = DefensePipeline(corpus, mock, policy=MetadataPolicy(reference_year=2026))
    poisoned = pipe.answer("What is inflation?", "Finance", LayerConfig("p", rag="poisoned", prompt="full"))
    assert poisoned.poisoned_context and "purchasing power of money" in poisoned.draft_answer
    verified = pipe.answer("What is inflation?", "Finance", LayerConfig("pv", rag="poisoned", prompt="full", verifier=True))
    assert verified.corrected and "purchasing power of money" not in verified.final_answer
    filtered = pipe.answer("What is inflation?", "Finance", LayerConfig("f", rag="filtered", prompt="full"))
    assert not filtered.poisoned_context and "purchasing power of money" not in filtered.draft_answer


def test_generator_model_map_per_alignment(corpus, mock):
    pipe = DefensePipeline(corpus, mock, generator_models={"base": "base-ckpt"})
    assert pipe.answer("q", "Finance", LayerConfig("b", alignment="base")).generation.model == "base-ckpt"
    assert pipe.answer("q", "Finance", LayerConfig("a", alignment="aligned")).generation.model == "mock-model"


def test_judge_rejects_bad_output():
    class Bad:
        name = "bad"

        def chat(self, messages, **kw):
            from layered_defense import ChatResponse

            return ChatResponse("not json", 0.0, "m")

    with pytest.raises(ValueError, match="non-JSON"):
        judge_response(Bad(), "q", "a", "e", [])
