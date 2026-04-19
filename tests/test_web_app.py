"""Tests for the Flask web application."""

import json
import os

import pytest

from rhyme_web.app import create_app


@pytest.fixture
def app(tmp_path):
    # Use SQLite in temp dir for tests
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
    {"id": f"INC-{i:03d}", "summary": f"Test incident {i} with unique details about scenario {i}", "timestamp": f"2024-01-{(i%28)+1:02d}T10:00:00Z", "severity": "SEV1", "service": f"service-{i}"}
    for i in range(8)
])


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "ok"


def test_index_empty(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"No labeling sessions" in resp.data


def test_import_page_get(client):
    resp = client.get("/import")
    assert resp.status_code == 200
    assert b"Import Incidents" in resp.data


def test_import_empty_submission(client):
    resp = client.post("/import", data={})
    assert resp.status_code == 200
    assert b"No data provided" in resp.data


def test_import_invalid_json(client):
    resp = client.post("/import", data={"json_text": "not json"})
    assert resp.status_code == 200
    assert b"Import error" in resp.data


def test_import_too_few_incidents(client):
    resp = client.post("/import", data={"json_text": json.dumps([
        {"id": "1", "summary": "test", "timestamp": "2024-01-01T00:00:00Z"},
    ])})
    assert resp.status_code == 200
    assert b"at least 5" in resp.data


def test_import_success(client):
    resp = client.post("/import", data={"json_text": SAMPLE_INCIDENTS}, follow_redirects=False)
    assert resp.status_code == 302
    assert "/session/" in resp.headers["Location"]


def test_session_not_found(client):
    resp = client.get("/session/nonexistent", follow_redirects=True)
    assert resp.status_code == 200
    assert b"Session not found" in resp.data


def test_full_flow(client):
    # Import
    resp = client.post("/import", data={"json_text": SAMPLE_INCIDENTS}, follow_redirects=True)
    assert resp.status_code == 200
    assert b"Incident A" in resp.data

    # Get session ID from the URL
    session_id = resp.request.path.split("/")[-1]

    # Get session API data
    api_resp = client.get(f"/api/session/{session_id}")
    assert api_resp.status_code == 200
    session_data = api_resp.get_json()
    assert len(session_data["pairs"]) > 0
    pair_id = session_data["pairs"][0]["pair_id"]

    # Submit a label
    resp = client.post(f"/session/{session_id}/label", data={
        "pair_id": pair_id,
        "judgment": "yes",
        "labeler_id": "test_user",
        "notes": "same proximal cause",
    }, follow_redirects=True)
    assert resp.status_code == 200

    # Check dashboard
    resp = client.get(f"/dashboard/{session_id}")
    assert resp.status_code == 200
    assert b"Pairs labeled" in resp.data

    # Check export
    resp = client.get(f"/api/export/{session_id}")
    assert resp.status_code == 200
    export = resp.get_json()
    assert export["session"]["session_id"] == session_id
    assert len(export["session"]["labels"]) == 1
    assert export["report"] is not None
    assert export["report"]["labeled_pairs"] == 1


def test_index_shows_session_after_import(client):
    client.post("/import", data={"json_text": SAMPLE_INCIDENTS})
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"Continue" in resp.data or b"Dashboard" in resp.data


def test_label_page_redirects_when_done(client):
    # Import minimal corpus
    small = json.dumps([
        {"id": f"INC-{i}", "summary": f"s{i}", "timestamp": "2024-01-01T00:00:00Z"}
        for i in range(5)
    ])
    resp = client.post("/import", data={"json_text": small}, follow_redirects=True)
    session_id = resp.request.path.split("/")[-1]

    # Label all pairs
    api_resp = client.get(f"/api/session/{session_id}")
    session_data = api_resp.get_json()
    for pair in session_data["pairs"]:
        client.post(f"/session/{session_id}/label", data={
            "pair_id": pair["pair_id"],
            "judgment": "no",
            "labeler_id": "test",
        })

    # Labeling page should redirect to dashboard
    resp = client.get(f"/session/{session_id}")
    assert resp.status_code == 302
    assert "/dashboard/" in resp.headers["Location"]
