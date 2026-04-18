"""Tests for corpus generation."""

from sifter_bench.generator import generate_corpus, generate_incident
from sifter_bench.taxonomy import CauseClass, ConfusabilityTier
import random


def test_generate_corpus_correct_size():
    corpus = generate_corpus(incidents_per_class=3, seed=1)
    assert len(corpus.incidents) == 3 * len(CauseClass)


def test_generate_corpus_all_classes_represented():
    corpus = generate_corpus(incidents_per_class=5, seed=2)
    classes = {inc.labels.cause_class for inc in corpus.incidents}
    assert classes == set(CauseClass)


def test_generate_corpus_deterministic():
    c1 = generate_corpus(incidents_per_class=3, seed=42)
    c2 = generate_corpus(incidents_per_class=3, seed=42)
    ids1 = [inc.payload.incident_id for inc in c1.incidents]
    ids2 = [inc.payload.incident_id for inc in c2.incidents]
    assert ids1 == ids2


def test_generate_corpus_different_seeds_differ():
    c1 = generate_corpus(incidents_per_class=3, seed=1)
    c2 = generate_corpus(incidents_per_class=3, seed=2)
    summaries1 = [inc.payload.summary for inc in c1.incidents]
    summaries2 = [inc.payload.summary for inc in c2.incidents]
    assert summaries1 != summaries2


def test_incident_has_required_payload_fields():
    rng = random.Random(42)
    inc = generate_incident(CauseClass.RETRY_STORM, 0, rng)
    p = inc.payload
    assert p.incident_id
    assert p.summary
    assert 5 <= len(p.alerts) <= 15
    assert 20 <= len(p.log_lines) <= 100
    assert p.topology.nodes
    assert p.incident_start


def test_incident_has_required_label_fields():
    rng = random.Random(42)
    inc = generate_incident(CauseClass.MEMORY_LEAK, 0, rng)
    labels = inc.labels
    assert labels.cause_class == CauseClass.MEMORY_LEAK
    assert labels.confusability_tier in ConfusabilityTier
    assert labels.remediation_canonical
    assert labels.remediation_masks_symptom
    assert labels.fingerprint


def test_deploy_classes_have_deploy_type():
    rng = random.Random(42)
    code = generate_incident(CauseClass.CODE_REGRESSION, 0, rng)
    assert code.labels.fingerprint.deploy_type == "code"

    config = generate_incident(CauseClass.CONFIG_REGRESSION, 0, rng)
    assert config.labels.fingerprint.deploy_type == "config"


def test_non_deploy_classes_have_no_deploy_type():
    rng = random.Random(42)
    inc = generate_incident(CauseClass.MEMORY_LEAK, 0, rng)
    assert inc.labels.fingerprint.deploy_type is None


def test_connection_pool_has_db_metrics():
    rng = random.Random(42)
    inc = generate_incident(CauseClass.CONNECTION_POOL_EXHAUSTION, 0, rng)
    fp = inc.labels.fingerprint
    assert fp.db_query_time_ms is not None
    assert fp.db_connection_pool_utilization is not None
    assert fp.db_connection_pool_utilization >= 0.95  # pool should be maxed


def test_downstream_slowdown_has_db_metrics():
    rng = random.Random(42)
    inc = generate_incident(CauseClass.DOWNSTREAM_SLOWDOWN, 0, rng)
    fp = inc.labels.fingerprint
    assert fp.db_query_time_ms is not None
    assert fp.db_query_time_ms >= 500  # slow queries


def test_corpus_is_shuffled():
    corpus = generate_corpus(incidents_per_class=5, seed=42)
    classes = [inc.labels.cause_class for inc in corpus.incidents]
    # If not shuffled, first N would all be the same class
    first_5_classes = set(classes[:5])
    assert len(first_5_classes) > 1, "Corpus appears not shuffled"


def test_incident_ids_are_unique():
    corpus = generate_corpus(incidents_per_class=5, seed=42)
    ids = [inc.payload.incident_id for inc in corpus.incidents]
    assert len(ids) == len(set(ids))
