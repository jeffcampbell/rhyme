"""Tests for the multi-model comparison feature."""

import json
import os

import pytest

from rhyme_web.app import create_app


@pytest.fixture
def app(tmp_path):
    db_path = tmp_path / "test.db"
    os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"
    app = create_app(data_dir=str(tmp_path))
    app.config["TESTING"] = True
    yield app
    os.environ.pop("DATABASE_URL", None)


@pytest.fixture
def client(app):
    return app.test_client()


SAMPLE_INCIDENTS = json.dumps([
    {
        "id": f"INC-{i:03d}",
        "summary": f"Test incident {i} about scenario {i}",
        "timestamp": f"2024-01-{(i % 28) + 1:02d}T10:00:00Z",
        "severity": "SEV1",
        "service": f"service-{i}",
    }
    for i in range(8)
])


def _create_session(client):
    """Import incidents and return session_id and pair_ids."""
    resp = client.post("/import", data={"json_text": SAMPLE_INCIDENTS}, follow_redirects=True)
    session_id = resp.request.path.split("/")[-1]
    api_resp = client.get(f"/api/session/{session_id}")
    pairs = api_resp.get_json()["pairs"]
    return session_id, [p["pair_id"] for p in pairs]


def _label_pairs(client, session_id, pair_ids, n=10):
    """Submit labels for the first n pairs."""
    judgments = ["yes", "no", "no", "yes", "maybe", "no", "yes", "no", "yes", "no"]
    for i, pair_id in enumerate(pair_ids[:n]):
        client.post(f"/session/{session_id}/label", data={
            "pair_id": pair_id,
            "judgment": judgments[i % len(judgments)],
            "labeler_id": "test-sre",
        })


# --- Score upload API tests ---

class TestScoreUploadAPI:
    def test_upload_scores_success(self, client):
        session_id, pair_ids = _create_session(client)
        scores = [{"pair_id": pid, "confidence": 0.8} for pid in pair_ids[:5]]
        resp = client.post(
            f"/api/scores/{session_id}",
            json={"model_name": "test-model", "scores": scores},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["saved"] == 5
        assert data["model_name"] == "test-model"

    def test_upload_scores_session_not_found(self, client):
        resp = client.post(
            "/api/scores/nonexistent",
            json={"model_name": "m", "scores": []},
        )
        assert resp.status_code == 404

    def test_upload_scores_missing_fields(self, client):
        session_id, _ = _create_session(client)
        resp = client.post(f"/api/scores/{session_id}", json={"scores": []})
        assert resp.status_code == 400

    def test_upload_scores_empty_body(self, client):
        session_id, _ = _create_session(client)
        resp = client.post(
            f"/api/scores/{session_id}",
            data="",
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_upload_scores_invalid_pair_ids_skipped(self, client):
        session_id, pair_ids = _create_session(client)
        scores = [
            {"pair_id": pair_ids[0], "confidence": 0.9},
            {"pair_id": "fake-pair-id", "confidence": 0.5},
        ]
        resp = client.post(
            f"/api/scores/{session_id}",
            json={"model_name": "m", "scores": scores},
        )
        assert resp.get_json()["saved"] == 1

    def test_upload_scores_clamps_confidence(self, client):
        session_id, pair_ids = _create_session(client)
        scores = [
            {"pair_id": pair_ids[0], "confidence": 1.5},
            {"pair_id": pair_ids[1], "confidence": -0.3},
        ]
        resp = client.post(
            f"/api/scores/{session_id}",
            json={"model_name": "m", "scores": scores},
        )
        assert resp.get_json()["saved"] == 2

    def test_upload_scores_invalid_confidence_skipped(self, client):
        session_id, pair_ids = _create_session(client)
        scores = [
            {"pair_id": pair_ids[0], "confidence": "not-a-number"},
            {"pair_id": pair_ids[1], "confidence": 0.7},
        ]
        resp = client.post(
            f"/api/scores/{session_id}",
            json={"model_name": "m", "scores": scores},
        )
        assert resp.get_json()["saved"] == 1

    def test_upload_scores_upsert_replaces_existing(self, client):
        session_id, pair_ids = _create_session(client)
        scores = [{"pair_id": pair_ids[0], "confidence": 0.3}]
        client.post(
            f"/api/scores/{session_id}",
            json={"model_name": "m", "scores": scores},
        )
        # Upload again with different confidence
        scores = [{"pair_id": pair_ids[0], "confidence": 0.9}]
        resp = client.post(
            f"/api/scores/{session_id}",
            json={"model_name": "m", "scores": scores},
        )
        assert resp.get_json()["saved"] == 1

        # Verify via models API
        resp = client.get(f"/api/models/{session_id}")
        models = resp.get_json()["models"]
        assert len(models) == 1
        assert models[0]["score_count"] == 1

    def test_upload_multiple_models(self, client):
        session_id, pair_ids = _create_session(client)
        for model in ["model-a", "model-b", "model-c"]:
            scores = [{"pair_id": pid, "confidence": 0.5} for pid in pair_ids[:3]]
            client.post(
                f"/api/scores/{session_id}",
                json={"model_name": model, "scores": scores},
            )
        resp = client.get(f"/api/models/{session_id}")
        models = resp.get_json()["models"]
        assert len(models) == 3


# --- Models list API tests ---

class TestModelsListAPI:
    def test_list_models_empty(self, client):
        session_id, _ = _create_session(client)
        resp = client.get(f"/api/models/{session_id}")
        assert resp.status_code == 200
        assert resp.get_json()["models"] == []

    def test_list_models_with_scores(self, client):
        session_id, pair_ids = _create_session(client)
        client.post(
            f"/api/scores/{session_id}",
            json={"model_name": "gpt-4o", "scores": [
                {"pair_id": pair_ids[0], "confidence": 0.8},
                {"pair_id": pair_ids[1], "confidence": 0.6},
            ]},
        )
        resp = client.get(f"/api/models/{session_id}")
        models = resp.get_json()["models"]
        assert len(models) == 1
        assert models[0]["name"] == "gpt-4o"
        assert models[0]["score_count"] == 2


# --- Export API includes incidents ---

class TestExportAPI:
    def test_export_includes_incidents(self, client):
        session_id, _ = _create_session(client)
        resp = client.get(f"/api/export/{session_id}")
        data = resp.get_json()
        assert "incidents" in data
        assert len(data["incidents"]) == 8
        inc = data["incidents"][0]
        assert "id" in inc
        assert "summary" in inc
        assert "timestamp" in inc


# --- Dashboard with model comparison ---

class TestDashboardComparison:
    def test_dashboard_no_models_no_labels(self, client):
        session_id, _ = _create_session(client)
        resp = client.get(f"/dashboard/{session_id}")
        assert resp.status_code == 200
        assert b"No labels yet" in resp.data
        assert b"Evaluate a Model" in resp.data

    def test_dashboard_with_labels_no_models(self, client):
        session_id, pair_ids = _create_session(client)
        _label_pairs(client, session_id, pair_ids, n=5)
        resp = client.get(f"/dashboard/{session_id}")
        assert resp.status_code == 200
        assert b"Pairs labeled" in resp.data
        # Should not show model comparison table
        assert b"Model Comparison vs Human Labels" not in resp.data

    def test_dashboard_with_models_and_labels(self, client):
        session_id, pair_ids = _create_session(client)
        _label_pairs(client, session_id, pair_ids, n=10)

        # Upload two models
        scores_a = [{"pair_id": pid, "confidence": 0.85} for pid in pair_ids]
        scores_b = [{"pair_id": pid, "confidence": 0.3} for pid in pair_ids]
        client.post(
            f"/api/scores/{session_id}",
            json={"model_name": "claude-haiku", "scores": scores_a},
        )
        client.post(
            f"/api/scores/{session_id}",
            json={"model_name": "gpt-4o-mini", "scores": scores_b},
        )

        resp = client.get(f"/dashboard/{session_id}")
        assert resp.status_code == 200
        assert b"Model Comparison vs Human Labels" in resp.data
        assert b"claude-haiku" in resp.data
        assert b"gpt-4o-mini" in resp.data
        assert b"Best model by F1" in resp.data

    def test_dashboard_cli_instructions_present(self, client):
        session_id, _ = _create_session(client)
        resp = client.get(f"/dashboard/{session_id}")
        assert resp.status_code == 200
        assert b"rhyme-eval" in resp.data
        assert session_id.encode() in resp.data


# --- Scorer multi-model tests ---

class TestScorerMultiModel:
    def test_score_model_against_humans(self):
        from rhyme_web.models import IncidentPair, PairLabel, LabelingSession
        from rhyme_web.scorer_human import score_model_against_humans

        pairs = [
            IncidentPair(pair_id=f"p{i}", incident_a_id=f"a{i}", incident_b_id=f"b{i}", sampling_bucket="random")
            for i in range(10)
        ]
        labels = [
            PairLabel(pair_id="p0", labeler_id="sre1", judgment="yes"),
            PairLabel(pair_id="p1", labeler_id="sre1", judgment="yes"),
            PairLabel(pair_id="p2", labeler_id="sre1", judgment="no"),
            PairLabel(pair_id="p3", labeler_id="sre1", judgment="no"),
            PairLabel(pair_id="p4", labeler_id="sre1", judgment="no"),
        ]
        session = LabelingSession(session_id="test", corpus_path="", pairs=pairs, labels=labels)

        # Model that's confident on "yes" pairs and not on "no" pairs
        good_model = {"p0": 0.9, "p1": 0.85, "p2": 0.2, "p3": 0.1, "p4": 0.15}
        report = score_model_against_humans(session, good_model)
        assert report.precision_at_50 == 1.0  # both positives are above 0.5
        assert report.recall_at_50 == 1.0

        # Model that's confident on wrong pairs
        bad_model = {"p0": 0.1, "p1": 0.2, "p2": 0.9, "p3": 0.85, "p4": 0.8}
        report = score_model_against_humans(session, bad_model)
        assert report.precision_at_50 == 0.0  # all positives are negatives
        assert report.high_conf_disagreements == 3

    def test_score_model_missing_pairs_get_default(self):
        from rhyme_web.models import IncidentPair, PairLabel, LabelingSession
        from rhyme_web.scorer_human import score_model_against_humans

        pairs = [
            IncidentPair(pair_id="p0", incident_a_id="a0", incident_b_id="b0", sampling_bucket="random"),
            IncidentPair(pair_id="p1", incident_a_id="a1", incident_b_id="b1", sampling_bucket="random"),
        ]
        labels = [
            PairLabel(pair_id="p0", labeler_id="sre1", judgment="yes"),
            PairLabel(pair_id="p1", labeler_id="sre1", judgment="no"),
        ]
        session = LabelingSession(session_id="test", corpus_path="", pairs=pairs, labels=labels)

        # Only score one pair — other gets default 0.5
        report = score_model_against_humans(session, {"p0": 0.9})
        assert report.labeled_pairs == 2
        # p0: conf=0.9, human=yes -> TP at 0.5 threshold
        # p1: conf=0.5 (default), human=no -> FP at 0.5 threshold
        assert report.precision_at_50 == 0.5

    def test_score_model_empty_labels(self):
        from rhyme_web.models import IncidentPair, LabelingSession
        from rhyme_web.scorer_human import score_model_against_humans

        pairs = [
            IncidentPair(pair_id="p0", incident_a_id="a0", incident_b_id="b0", sampling_bucket="random"),
        ]
        session = LabelingSession(session_id="test", corpus_path="", pairs=pairs, labels=[])

        report = score_model_against_humans(session, {"p0": 0.9})
        assert report.labeled_pairs == 0
        assert report.precision_at_50 == 0
