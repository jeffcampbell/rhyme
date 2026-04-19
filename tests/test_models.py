"""Tests for data models serialization and validation."""

import json
from pathlib import Path

from rhyme_bench.models import (
    Alert,
    Corpus,
    Fingerprint,
    Incident,
    IncidentLabels,
    IncidentPayload,
    LogLine,
    QuerySet,
    RankedMatch,
    RemediationChoice,
    RemediationQuestion,
    RemediationQuestionSet,
    RetrievalResult,
    TokenUsage,
    TopologyEdge,
    TopologyFragment,
    TopologyNode,
)
from rhyme_bench.taxonomy import CauseClass, ConfusabilityTier


def test_token_usage_total():
    t = TokenUsage(input_tokens=100, output_tokens=50)
    assert t.total_tokens == 150


def test_corpus_roundtrip(small_corpus, tmp_path):
    path = tmp_path / "corpus.json"
    small_corpus.save(path)
    loaded = Corpus.load(path)
    assert len(loaded.incidents) == len(small_corpus.incidents)
    assert loaded.version == small_corpus.version
    assert loaded.incidents[0].payload.incident_id == small_corpus.incidents[0].payload.incident_id


def test_query_set_roundtrip(query_set_and_incidents, tmp_path):
    qs, _ = query_set_and_incidents
    path = tmp_path / "queries.json"
    qs.save(path)
    loaded = QuerySet.load(path)
    assert len(loaded.queries) == len(qs.queries)


def test_remediation_question_set_roundtrip(remediation_questions, tmp_path):
    path = tmp_path / "rem.json"
    remediation_questions.save(path)
    loaded = RemediationQuestionSet.load(path)
    assert len(loaded.questions) == len(remediation_questions.questions)


def test_corpus_payloads_exclude_labels(small_corpus):
    payloads = small_corpus.payloads()
    for p in payloads:
        assert isinstance(p, IncidentPayload)
        # Payloads should not have any label fields
        d = p.model_dump()
        assert "cause_class" not in d
        assert "fingerprint" not in d


def test_retrieval_result_with_token_usage():
    r = RetrievalResult(
        query_id="q1",
        ranked_matches=[RankedMatch(incident_id="i1", confidence=0.9)],
        token_usage=TokenUsage(input_tokens=500, output_tokens=100),
    )
    d = r.model_dump()
    assert d["token_usage"]["input_tokens"] == 500
    loaded = RetrievalResult.model_validate(d)
    assert loaded.token_usage.total_tokens == 600


def test_retrieval_result_without_token_usage():
    r = RetrievalResult(
        query_id="q1",
        ranked_matches=[RankedMatch(incident_id="i1", confidence=0.9)],
    )
    d = r.model_dump()
    assert d["token_usage"] is None
    loaded = RetrievalResult.model_validate(d)
    assert loaded.token_usage is None


def test_fingerprint_optional_fields():
    fp = Fingerprint(
        latency_shift="bimodal",
        error_rate_pattern="step_increase",
        traffic_correlation="uncorrelated",
        origin_service="svc-a",
        affected_services=["svc-a"],
        topology_pattern="single_service",
        onset_shape="step",
        duration_pattern="persistent",
        deploy_correlation=False,
        amplification_ratio=1.0,
    )
    assert fp.db_query_time_ms is None
    assert fp.deploy_type is None

    fp2 = Fingerprint(
        latency_shift="bimodal",
        error_rate_pattern="step_increase",
        traffic_correlation="uncorrelated",
        origin_service="svc-a",
        affected_services=["svc-a"],
        topology_pattern="single_service",
        onset_shape="step",
        duration_pattern="persistent",
        deploy_correlation=True,
        amplification_ratio=1.0,
        db_query_time_ms=500.0,
        db_connection_pool_utilization=0.95,
        deploy_type="code",
    )
    assert fp2.db_query_time_ms == 500.0
    assert fp2.deploy_type == "code"
