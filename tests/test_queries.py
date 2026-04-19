"""Tests for query set construction and remediation questions."""

from rhyme_bench.queries import (
    build_query_set,
    build_private_slice,
    build_remediation_questions,
)
from rhyme_bench.taxonomy import CauseClass, ConfusabilityTier


def test_query_set_size(small_corpus):
    qs, qi = build_query_set(small_corpus, total_queries=20, seed=1)
    assert len(qs.queries) == 20
    assert len(qi) == 20


def test_query_ids_match(small_corpus):
    qs, qi = build_query_set(small_corpus, total_queries=20, seed=1)
    qs_ids = {q.query_id for q in qs.queries}
    qi_ids = {inc.payload.incident_id for inc in qi}
    assert qs_ids == qi_ids


def test_queries_not_in_own_correct_matches(small_corpus):
    qs, qi = build_query_set(small_corpus, total_queries=10, seed=1)
    for q in qs.queries:
        assert q.query_id not in q.correct_match_ids


def test_correct_matches_share_cause_class(small_corpus):
    qs, qi = build_query_set(small_corpus, total_queries=10, seed=1)
    inc_map = {inc.payload.incident_id: inc for inc in small_corpus.incidents}
    for q in qs.queries:
        query_class = inc_map[q.query_id].labels.cause_class
        for match_id in q.correct_match_ids:
            assert inc_map[match_id].labels.cause_class == query_class


def test_tempting_wrong_matches_differ_in_class(small_corpus):
    qs, qi = build_query_set(small_corpus, total_queries=10, seed=1)
    inc_map = {inc.payload.incident_id: inc for inc in small_corpus.incidents}
    for q in qs.queries:
        query_class = inc_map[q.query_id].labels.cause_class
        for match_id in q.tempting_wrong_ids:
            assert inc_map[match_id].labels.cause_class != query_class


def test_tier_stratification(small_corpus):
    qs, qi = build_query_set(small_corpus, total_queries=40, seed=1)
    tier_counts = {}
    for q in qs.queries:
        tier_counts[q.confusability_tier] = tier_counts.get(q.confusability_tier, 0) + 1
    # Should have all three tiers represented
    assert len(tier_counts) >= 2  # at least 2 tiers with 5 incidents/class


def test_private_slice_no_overlap(small_corpus):
    qs, qi = build_query_set(small_corpus, total_queries=20, seed=1)
    public_ids = {q.query_id for q in qs.queries}

    private_qs, private_qi = build_private_slice(small_corpus, public_ids, total_queries=10)
    private_ids = {q.query_id for q in private_qs.queries}

    assert public_ids.isdisjoint(private_ids), "Private slice overlaps with public queries"


def test_remediation_questions_count(query_set_and_incidents):
    _, qi = query_set_and_incidents
    rqs = build_remediation_questions(qi)
    assert len(rqs.questions) == len(qi)


def test_remediation_questions_have_5_choices(query_set_and_incidents):
    _, qi = query_set_and_incidents
    rqs = build_remediation_questions(qi)
    for q in rqs.questions:
        assert len(q.choices) == 5
        labels = {c.label for c in q.choices}
        assert labels == {"A", "B", "C", "D", "E"}


def test_remediation_questions_have_correct_answer(query_set_and_incidents):
    _, qi = query_set_and_incidents
    rqs = build_remediation_questions(qi)
    for q in rqs.questions:
        assert q.correct_label in {"A", "B", "C", "D", "E"}
        correct = [c for c in q.choices if c.label == q.correct_label]
        assert len(correct) == 1
        assert correct[0].grade == "fixed"


def test_remediation_questions_have_masks_symptom(query_set_and_incidents):
    _, qi = query_set_and_incidents
    rqs = build_remediation_questions(qi)
    for q in rqs.questions:
        grades = {c.grade for c in q.choices}
        assert "masks_symptom" in grades


def test_remediation_questions_deterministic(query_set_and_incidents):
    _, qi = query_set_and_incidents
    rqs1 = build_remediation_questions(qi, seed=100)
    rqs2 = build_remediation_questions(qi, seed=100)
    for q1, q2 in zip(rqs1.questions, rqs2.questions):
        assert q1.correct_label == q2.correct_label
        assert [c.label for c in q1.choices] == [c.label for c in q2.choices]
