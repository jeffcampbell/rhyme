"""Tests for pair sampling."""

from sifter_web.models import CorrelationScore, OrgCorpus, OrgIncident
from sifter_web.pair_sampler import sample_pairs


def _make_corpus(n: int) -> OrgCorpus:
    return OrgCorpus(incidents=[
        OrgIncident(id=f"INC-{i:03d}", summary=f"Incident {i}", timestamp=f"2024-01-{(i%28)+1:02d}T10:00:00Z")
        for i in range(n)
    ])


def test_sample_pairs_correct_count():
    corpus = _make_corpus(20)
    session = sample_pairs(corpus, total_pairs=50)
    assert len(session.pairs) == 50


def test_sample_pairs_capped_at_total_combinations():
    corpus = _make_corpus(5)  # C(5,2) = 10 pairs
    session = sample_pairs(corpus, total_pairs=50)
    assert len(session.pairs) == 10


def test_sample_pairs_no_self_pairs():
    corpus = _make_corpus(10)
    session = sample_pairs(corpus, total_pairs=20)
    for pair in session.pairs:
        assert pair.incident_a_id != pair.incident_b_id


def test_sample_pairs_unique():
    corpus = _make_corpus(15)
    session = sample_pairs(corpus, total_pairs=30)
    pair_keys = [(p.incident_a_id, p.incident_b_id) for p in session.pairs]
    assert len(pair_keys) == len(set(pair_keys))


def test_sample_pairs_all_random_without_scores():
    corpus = _make_corpus(10)
    session = sample_pairs(corpus, total_pairs=20)
    for pair in session.pairs:
        assert pair.sampling_bucket == "random"
        assert pair.model_confidence is None


def test_sample_pairs_stratified_with_scores():
    corpus = _make_corpus(20)
    scores = [
        CorrelationScore(incident_a_id=f"INC-{i:03d}", incident_b_id=f"INC-{j:03d}", confidence=i*j/400)
        for i in range(20) for j in range(i+1, 20)
    ]
    session = sample_pairs(corpus, model_scores=scores, total_pairs=40)

    buckets = {p.sampling_bucket for p in session.pairs}
    assert "high_confidence" in buckets
    assert "low_confidence" in buckets
    assert "random" in buckets


def test_sample_pairs_deterministic():
    corpus = _make_corpus(15)
    s1 = sample_pairs(corpus, total_pairs=20, seed=42)
    s2 = sample_pairs(corpus, total_pairs=20, seed=42)
    ids1 = [(p.incident_a_id, p.incident_b_id) for p in s1.pairs]
    ids2 = [(p.incident_a_id, p.incident_b_id) for p in s2.pairs]
    assert ids1 == ids2


def test_sample_pairs_different_seeds_differ():
    corpus = _make_corpus(15)
    s1 = sample_pairs(corpus, total_pairs=20, seed=1)
    s2 = sample_pairs(corpus, total_pairs=20, seed=2)
    ids1 = [(p.incident_a_id, p.incident_b_id) for p in s1.pairs]
    ids2 = [(p.incident_a_id, p.incident_b_id) for p in s2.pairs]
    assert ids1 != ids2
