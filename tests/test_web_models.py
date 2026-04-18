"""Tests for web tool data models and import."""

import json
from pathlib import Path

import pytest

from sifter_web.models import (
    OrgCorpus,
    OrgIncident,
    LabelingSession,
    PairLabel,
    IncidentPair,
    import_incidents,
    parse_incidents,
)


@pytest.fixture
def sample_incidents_json(tmp_path):
    data = {
        "incidents": [
            {"id": f"INC-{i:03d}", "summary": f"Incident {i} summary", "timestamp": f"2024-01-{i+1:02d}T10:00:00Z"}
            for i in range(10)
        ]
    }
    path = tmp_path / "incidents.json"
    path.write_text(json.dumps(data))
    return path


@pytest.fixture
def sample_bare_array_json(tmp_path):
    data = [
        {"id": f"INC-{i:03d}", "summary": f"Incident {i}", "timestamp": f"2024-01-{i+1:02d}T10:00:00Z"}
        for i in range(5)
    ]
    path = tmp_path / "incidents.json"
    path.write_text(json.dumps(data))
    return path


def test_import_incidents_object_format(sample_incidents_json):
    result = import_incidents(sample_incidents_json)
    assert len(result.corpus.incidents) == 10
    assert result.corpus.incidents[0].id == "INC-000"
    assert not result.has_errors


def test_import_incidents_bare_array(sample_bare_array_json):
    result = import_incidents(sample_bare_array_json)
    assert len(result.corpus.incidents) == 5
    assert not result.has_errors


def test_import_invalid_json(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"not_incidents": []}')
    with pytest.raises(ValueError, match="Expected"):
        import_incidents(path)


def test_parse_incidents_from_string():
    json_text = '[{"id": "1", "summary": "test", "timestamp": "2024-01-01T00:00:00Z"}]'
    result = parse_incidents(json_text)
    assert len(result.corpus.incidents) == 1
    assert result.corpus.incidents[0].id == "1"
    assert not result.has_errors


def test_org_incident_optional_fields():
    inc = OrgIncident(id="1", summary="test", timestamp="2024-01-01T00:00:00Z")
    assert inc.severity is None
    assert inc.service is None
    assert inc.metadata == {}


def test_org_incident_with_metadata():
    inc = OrgIncident(
        id="1", summary="test", timestamp="2024-01-01T00:00:00Z",
        severity="SEV1", service="api-gateway", metadata={"team": "platform"},
    )
    assert inc.metadata["team"] == "platform"


def test_org_corpus_roundtrip(tmp_path):
    corpus = OrgCorpus(incidents=[
        OrgIncident(id="1", summary="test1", timestamp="2024-01-01T00:00:00Z"),
        OrgIncident(id="2", summary="test2", timestamp="2024-01-02T00:00:00Z"),
    ])
    path = tmp_path / "corpus.json"
    corpus.save(path)
    loaded = OrgCorpus.load(path)
    assert len(loaded.incidents) == 2


def test_labeling_session_progress():
    session = LabelingSession(
        session_id="test",
        corpus_path="/tmp/test.json",
        pairs=[
            IncidentPair(pair_id="p1", incident_a_id="1", incident_b_id="2", sampling_bucket="random"),
            IncidentPair(pair_id="p2", incident_a_id="1", incident_b_id="3", sampling_bucket="random"),
            IncidentPair(pair_id="p3", incident_a_id="2", incident_b_id="3", sampling_bucket="random"),
        ],
        labels=[
            PairLabel(pair_id="p1", labeler_id="jeff", judgment="yes"),
        ],
    )
    labeled, total = session.progress
    assert labeled == 1
    assert total == 3


def test_labeling_session_unlabeled():
    session = LabelingSession(
        session_id="test",
        corpus_path="/tmp/test.json",
        pairs=[
            IncidentPair(pair_id="p1", incident_a_id="1", incident_b_id="2", sampling_bucket="random"),
            IncidentPair(pair_id="p2", incident_a_id="1", incident_b_id="3", sampling_bucket="random"),
        ],
        labels=[
            PairLabel(pair_id="p1", labeler_id="jeff", judgment="yes"),
        ],
    )
    unlabeled = session.unlabeled_pairs()
    assert len(unlabeled) == 1
    assert unlabeled[0].pair_id == "p2"


def test_labeling_session_roundtrip(tmp_path):
    session = LabelingSession(
        session_id="test",
        corpus_path="/tmp/test.json",
        pairs=[IncidentPair(pair_id="p1", incident_a_id="1", incident_b_id="2", sampling_bucket="random")],
        labels=[PairLabel(pair_id="p1", labeler_id="jeff", judgment="no", notes="different causes")],
    )
    path = tmp_path / "session.json"
    session.save(path)
    loaded = LabelingSession.load(path)
    assert loaded.session_id == "test"
    assert len(loaded.labels) == 1
    assert loaded.labels[0].judgment == "no"
