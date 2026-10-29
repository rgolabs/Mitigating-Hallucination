import pytest

from layered_defense import Document, format_prompt
from layered_defense.prompts import NEGATIVE_CONSTRAINTS_CLOSED_BOOK, NEGATIVE_CONSTRAINTS_WITH_CONTEXT, is_refusal

DOC = Document("DOC-1", "Key concept: water boils at 100 C.", "Science")


def test_bare_prompt_has_no_system_instruction():
    p = format_prompt("q?", [DOC], "bare")
    assert "System Instruction" not in p and "Source [DOC-1]" in p


def test_full_prompt_with_context_uses_document_constraints():
    p = format_prompt("q?", [DOC], "full")
    assert NEGATIVE_CONSTRAINTS_WITH_CONTEXT[0] in p and "Example 1" in p


def test_closed_book_negatives_do_not_force_refusal():
    p = format_prompt("q?", [], "full")
    assert "No context available." in p
    assert NEGATIVE_CONSTRAINTS_CLOSED_BOOK[0] in p
    assert "outside the supplied documents" not in p


def test_unknown_config_rejected():
    with pytest.raises(ValueError):
        format_prompt("q?", [], "nope")


def test_is_refusal():
    assert is_refusal("I do not know.")
    assert is_refusal("Based on the context, I do not know the answer.")
    assert not is_refusal("Water boils at 100 C.")
