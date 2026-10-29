from layered_defense import ADVERSARIAL_DOCS, DocCorpus, Document, MetadataPolicy
from layered_defense.corpus import tokenize


def test_tokenize_strips_punctuation_and_stopwords():
    assert tokenize("What is an emergency fund?") == ["emergency", "fund"]


def test_policy_recency_is_relative_to_reference_year():
    doc = Document("d", "x", "Finance", publication_year=2022, quality_score=0.9)
    assert MetadataPolicy(recency_years=5, reference_year=2026).accepts(doc, "Finance")
    assert not MetadataPolicy(recency_years=3, reference_year=2026).accepts(doc, "Finance")
    assert MetadataPolicy(recency_years=None, reference_year=2026).accepts(doc, "Finance")


def test_policy_rejects_off_domain_and_low_quality():
    policy = MetadataPolicy(reference_year=2026)
    good = Document("d", "x", "Finance", publication_year=2025, quality_score=0.9)
    assert policy.accepts(good, "Finance")
    assert not policy.accepts(good, "Legal")
    assert policy.accepts(good, None)
    assert not policy.accepts(Document("d", "x", "Finance", publication_year=2025, quality_score=0.5), "Finance")


def test_poisoned_index_surfaces_adversarial_docs_but_filter_removes_them(corpus):
    q, dom = "What is inflation?", "Finance"
    poisoned = corpus.retrieve(q, dom, include_adversarial=True, attack_strength=1.0)
    assert any(d.adversarial for d in poisoned)
    clean = corpus.retrieve(q, dom, include_adversarial=False)
    assert not any(d.adversarial for d in clean)
    filtered = corpus.retrieve(q, dom, include_adversarial=True, policy=MetadataPolicy(reference_year=2026))
    assert filtered and not any(d.adversarial for d in filtered)
    assert all(d.doc_domain == dom for d in filtered)


def test_attack_strength_zero_means_no_lift(corpus):
    docs = corpus.retrieve("What is a learning objective?", "Education", include_adversarial=True, attack_strength=0.0, top_k=3)
    assert not any(d.adversarial for d in docs)


def test_poison_rate_is_about_five_percent(corpus):
    assert 0.04 < corpus.poison_rate < 0.06
    assert len(ADVERSARIAL_DOCS) == 6


def test_from_records_roundtrip():
    c = DocCorpus.from_records([{"doc_id": "1", "doc_text": "Key concept: cats purr.", "doc_domain": "Pets", "publication_year": "2025", "quality_score": "0.8"}])
    assert c.docs[0].publication_year == 2025 and c.docs[0].quality_score == 0.8
