"""Tests for baseline adapters."""

from rhyme_bench.baselines import (
    BM25Baseline,
    RandomBaseline,
    TfidfBaseline,
    _payload_to_text,
)
from rhyme_bench.models import RemediationChoice


def test_random_baseline_returns_k(small_corpus):
    adapter = RandomBaseline(seed=1)
    payloads = small_corpus.payloads()
    query = payloads[0]
    candidates = payloads[1:]
    matches = adapter.retrieve(query, candidates, k=5)
    assert len(matches) == 5


def test_random_baseline_confidence_bounded(small_corpus):
    adapter = RandomBaseline(seed=1)
    payloads = small_corpus.payloads()
    matches = adapter.retrieve(payloads[0], payloads[1:], k=10)
    for m in matches:
        assert 0.0 <= m.confidence <= 1.0


def test_random_baseline_deterministic(small_corpus):
    payloads = small_corpus.payloads()
    a1 = RandomBaseline(seed=42)
    a2 = RandomBaseline(seed=42)
    m1 = a1.retrieve(payloads[0], payloads[1:], k=5)
    m2 = a2.retrieve(payloads[0], payloads[1:], k=5)
    assert [m.incident_id for m in m1] == [m.incident_id for m in m2]


def test_random_baseline_remediate():
    adapter = RandomBaseline(seed=1)
    choices = [
        RemediationChoice(label="A", text="Fix A", grade="fixed"),
        RemediationChoice(label="B", text="Fix B", grade="no_op"),
    ]
    label = adapter.remediate(None, [], choices)
    assert label in {"A", "B"}


def test_bm25_prefers_same_class(small_corpus):
    adapter = BM25Baseline()
    payloads = small_corpus.payloads()
    inc_map = {inc.payload.incident_id: inc for inc in small_corpus.incidents}

    query = payloads[0]
    query_class = inc_map[query.incident_id].labels.cause_class
    matches = adapter.retrieve(query, payloads[1:], k=10)

    # At least the top match should share the cause class
    top_class = inc_map[matches[0].incident_id].labels.cause_class
    assert top_class == query_class


def test_bm25_confidence_normalized(small_corpus):
    adapter = BM25Baseline()
    payloads = small_corpus.payloads()
    matches = adapter.retrieve(payloads[0], payloads[1:], k=10)
    for m in matches:
        assert 0.0 <= m.confidence <= 1.0
    # Top match should have highest confidence
    confs = [m.confidence for m in matches]
    assert confs == sorted(confs, reverse=True)


def test_tfidf_returns_results(small_corpus):
    adapter = TfidfBaseline()
    payloads = small_corpus.payloads()
    matches = adapter.retrieve(payloads[0], payloads[1:], k=5)
    assert len(matches) == 5
    for m in matches:
        assert 0.0 <= m.confidence <= 1.0


def test_payload_to_text_includes_all_fields(small_corpus):
    payload = small_corpus.payloads()[0]
    text = _payload_to_text(payload)
    assert payload.summary in text
    assert any(a.message in text for a in payload.alerts)
    assert any(l.message in text for l in payload.log_lines)
