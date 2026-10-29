import pytest

from layered_defense import DocCorpus, MockProvider
from layered_defense.cli import DATA


@pytest.fixture
def corpus() -> DocCorpus:
    return DocCorpus.from_csv(DATA / "documents_corpus.csv").inject_adversarial()


@pytest.fixture
def mock() -> MockProvider:
    return MockProvider()
