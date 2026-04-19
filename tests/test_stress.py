"""Stress tests for the web app: weird inputs, malformed data, edge cases."""

import json
import os

import pytest

from rhyme_web.app import create_app
from rhyme_web.models import parse_incidents


# ---------------------------------------------------------------------------
# parse_incidents stress tests
# ---------------------------------------------------------------------------

class TestParseIncidentsStress:
    """Test parse_incidents with various malformed inputs."""

    def test_completely_invalid_json(self):
        with pytest.raises(ValueError, match="Invalid JSON"):
            parse_incidents("not json at all {{{")

    def test_empty_string(self):
        with pytest.raises(ValueError, match="Invalid JSON"):
            parse_incidents("")

    def test_null_json(self):
        with pytest.raises(ValueError, match="Expected"):
            parse_incidents("null")

    def test_number_json(self):
        with pytest.raises(ValueError, match="Expected"):
            parse_incidents("42")

    def test_string_json(self):
        with pytest.raises(ValueError, match="Expected"):
            parse_incidents('"hello"')

    def test_empty_array(self):
        result = parse_incidents("[]")
        assert len(result.corpus.incidents) == 0
        assert not result.has_errors

    def test_empty_incidents_key(self):
        result = parse_incidents('{"incidents": []}')
        assert len(result.corpus.incidents) == 0

    def test_mixed_valid_and_invalid(self):
        data = json.dumps([
            {"id": "1", "summary": "good one", "timestamp": "2024-01-01T00:00:00Z"},
            {"id": "2"},  # missing summary and timestamp
            {"id": "3", "summary": "also good", "timestamp": "2024-01-02T00:00:00Z"},
            "not an object",
            {"summary": "no id", "timestamp": "2024-01-03T00:00:00Z"},  # missing id
        ])
        result = parse_incidents(data)
        assert len(result.corpus.incidents) == 2
        assert result.has_errors
        assert len(result.errors) == 3

    def test_missing_id(self):
        data = json.dumps([{"summary": "test", "timestamp": "2024-01-01T00:00:00Z"}])
        result = parse_incidents(data)
        assert len(result.corpus.incidents) == 0
        assert "missing required" in result.errors[0].lower()

    def test_missing_summary(self):
        data = json.dumps([{"id": "1", "timestamp": "2024-01-01T00:00:00Z"}])
        result = parse_incidents(data)
        assert len(result.corpus.incidents) == 0
        assert result.has_errors

    def test_missing_timestamp(self):
        data = json.dumps([{"id": "1", "summary": "test"}])
        result = parse_incidents(data)
        assert len(result.corpus.incidents) == 0
        assert result.has_errors

    def test_empty_string_fields(self):
        data = json.dumps([{"id": "", "summary": "", "timestamp": ""}])
        result = parse_incidents(data)
        assert len(result.corpus.incidents) == 0
        assert result.has_errors

    def test_null_fields(self):
        data = json.dumps([{"id": None, "summary": None, "timestamp": None}])
        result = parse_incidents(data)
        assert len(result.corpus.incidents) == 0
        assert result.has_errors

    def test_numeric_id(self):
        """IDs should be coerced to strings."""
        data = json.dumps([{"id": 12345, "summary": "test", "timestamp": "2024-01-01T00:00:00Z"}])
        result = parse_incidents(data)
        assert len(result.corpus.incidents) == 1
        assert result.corpus.incidents[0].id == "12345"

    def test_very_long_summary(self):
        long_text = "x" * 100_000
        data = json.dumps([{"id": "1", "summary": long_text, "timestamp": "2024-01-01T00:00:00Z"}])
        result = parse_incidents(data)
        assert len(result.corpus.incidents) == 1
        assert len(result.corpus.incidents[0].summary) == 100_000

    def test_unicode_summary(self):
        data = json.dumps([{
            "id": "1",
            "summary": "Сервер упал. 서버가 다운되었습니다. サーバーがダウンしました。 🔥💀🚨",
            "timestamp": "2024-01-01T00:00:00Z",
        }])
        result = parse_incidents(data)
        assert len(result.corpus.incidents) == 1
        assert "🔥" in result.corpus.incidents[0].summary

    def test_html_in_summary(self):
        data = json.dumps([{
            "id": "1",
            "summary": '<script>alert("xss")</script><b>bold</b>',
            "timestamp": "2024-01-01T00:00:00Z",
        }])
        result = parse_incidents(data)
        assert len(result.corpus.incidents) == 1
        # Should store as-is (Jinja2 auto-escapes on render)
        assert "<script>" in result.corpus.incidents[0].summary

    def test_newlines_in_summary(self):
        data = json.dumps([{
            "id": "1",
            "summary": "Line 1\nLine 2\n\nLine 4\ttabbed",
            "timestamp": "2024-01-01T00:00:00Z",
        }])
        result = parse_incidents(data)
        assert len(result.corpus.incidents) == 1

    def test_special_characters_in_id(self):
        data = json.dumps([{
            "id": "INC-2024/001 (prod)",
            "summary": "test",
            "timestamp": "2024-01-01T00:00:00Z",
        }])
        result = parse_incidents(data)
        assert len(result.corpus.incidents) == 1

    def test_malformed_timestamp(self):
        """Non-ISO timestamps should still be accepted (string field, not validated)."""
        data = json.dumps([{
            "id": "1",
            "summary": "test",
            "timestamp": "last tuesday around 3pm",
        }])
        result = parse_incidents(data)
        assert len(result.corpus.incidents) == 1

    def test_extra_unknown_fields(self):
        """Extra fields should be ignored, not cause errors."""
        data = json.dumps([{
            "id": "1",
            "summary": "test",
            "timestamp": "2024-01-01T00:00:00Z",
            "unknown_field": "whatever",
            "nested": {"deep": True},
        }])
        result = parse_incidents(data)
        assert len(result.corpus.incidents) == 1
        assert not result.has_errors

    def test_duplicate_ids(self):
        """Duplicate IDs should be allowed at parse time (DB constraint handles uniqueness)."""
        data = json.dumps([
            {"id": "SAME", "summary": "first", "timestamp": "2024-01-01T00:00:00Z"},
            {"id": "SAME", "summary": "second", "timestamp": "2024-01-02T00:00:00Z"},
        ])
        result = parse_incidents(data)
        assert len(result.corpus.incidents) == 2

    def test_severity_truncated(self):
        data = json.dumps([{
            "id": "1",
            "summary": "test",
            "timestamp": "2024-01-01T00:00:00Z",
            "severity": "A" * 1000,
        }])
        result = parse_incidents(data)
        assert len(result.corpus.incidents) == 1
        assert len(result.corpus.incidents[0].severity) == 32

    def test_url_field(self):
        data = json.dumps([{
            "id": "1",
            "summary": "test",
            "timestamp": "2024-01-01T00:00:00Z",
            "url": "https://incidents.example.com/INC-001",
        }])
        result = parse_incidents(data)
        assert result.corpus.incidents[0].url == "https://incidents.example.com/INC-001"

    def test_array_of_non_objects(self):
        data = json.dumps([1, 2, "three", None, True])
        result = parse_incidents(data)
        assert len(result.corpus.incidents) == 0
        assert len(result.errors) == 5

    def test_incidents_key_not_array(self):
        with pytest.raises(ValueError, match="must be an array"):
            parse_incidents('{"incidents": "not an array"}')

    def test_large_batch(self):
        """1000 incidents should parse fine."""
        items = [
            {"id": f"INC-{i}", "summary": f"Incident {i} summary", "timestamp": "2024-01-01T00:00:00Z"}
            for i in range(1000)
        ]
        result = parse_incidents(json.dumps(items))
        assert len(result.corpus.incidents) == 1000
        assert not result.has_errors


# ---------------------------------------------------------------------------
# Web app stress tests
# ---------------------------------------------------------------------------

@pytest.fixture
def app(tmp_path):
    db_path = tmp_path / "stress.db"
    os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"
    app = create_app(data_dir=str(tmp_path))
    app.config["TESTING"] = True
    yield app
    os.environ.pop("DATABASE_URL", None)


@pytest.fixture
def client(app):
    return app.test_client()


def _make_incidents(n):
    return json.dumps([
        {"id": f"INC-{i}", "summary": f"Incident {i}", "timestamp": f"2024-01-{(i%28)+1:02d}T00:00:00Z"}
        for i in range(n)
    ])


class TestWebAppStress:

    def test_import_partial_with_errors(self, client):
        data = json.dumps([
            {"id": "1", "summary": "good", "timestamp": "2024-01-01T00:00:00Z"},
            {"id": "2"},  # bad
            {"id": "3", "summary": "good", "timestamp": "2024-01-02T00:00:00Z"},
            {"id": "4", "summary": "good", "timestamp": "2024-01-03T00:00:00Z"},
            {"id": "5", "summary": "good", "timestamp": "2024-01-04T00:00:00Z"},
            {"id": "6", "summary": "good", "timestamp": "2024-01-05T00:00:00Z"},
        ])
        resp = client.post("/import", data={"json_text": data, "session_name": "partial"}, follow_redirects=True)
        assert resp.status_code == 200
        assert b"skipped" in resp.data

    def test_import_all_invalid_shows_error(self, client):
        data = json.dumps([{"bad": True}, {"also": "bad"}])
        resp = client.post("/import", data={"json_text": data})
        assert resp.status_code == 200
        assert b"valid incident" in resp.data or b"at least 5" in resp.data

    def test_unicode_in_session_name(self, client):
        resp = client.post("/import", data={
            "json_text": _make_incidents(6),
            "session_name": "Тест сессия 🇯🇵 日本語テスト"
        }, follow_redirects=True)
        assert resp.status_code == 200

    def test_html_injection_in_summary(self, client):
        data = json.dumps([
            {"id": f"INC-{i}", "summary": f'<img src=x onerror=alert({i})>', "timestamp": "2024-01-01T00:00:00Z"}
            for i in range(6)
        ])
        resp = client.post("/import", data={"json_text": data}, follow_redirects=True)
        assert resp.status_code == 200
        # Jinja2 auto-escapes — raw HTML should not appear
        assert b"<img src=x" not in resp.data
        assert b"&lt;img" in resp.data

    def test_very_long_summary_in_ui(self, client):
        data = json.dumps([
            {"id": f"INC-{i}", "summary": f"word " * 2000, "timestamp": "2024-01-01T00:00:00Z"}
            for i in range(6)
        ])
        resp = client.post("/import", data={"json_text": data}, follow_redirects=True)
        assert resp.status_code == 200
        # Should have show-more toggle
        assert b"show-more" in resp.data.lower() or b"Show more" in resp.data

    def test_invalid_judgment_rejected(self, client):
        client.post("/import", data={"json_text": _make_incidents(6)})
        resp = client.get("/")
        # Get session ID
        session_id = resp.data.decode().split("/session/")[1].split('"')[0]

        api = client.get(f"/api/session/{session_id}")
        pair_id = api.get_json()["pairs"][0]["pair_id"]

        # Submit invalid judgment
        resp = client.post(f"/session/{session_id}/label", data={
            "pair_id": pair_id,
            "judgment": "definitely",
        }, follow_redirects=True)
        assert resp.status_code == 200
        # Should not have been saved
        api2 = client.get(f"/api/session/{session_id}")
        assert len(api2.get_json()["labels"]) == 0

    def test_empty_judgment_rejected(self, client):
        client.post("/import", data={"json_text": _make_incidents(6)})
        resp = client.get("/")
        session_id = resp.data.decode().split("/session/")[1].split('"')[0]
        api = client.get(f"/api/session/{session_id}")
        pair_id = api.get_json()["pairs"][0]["pair_id"]

        resp = client.post(f"/session/{session_id}/label", data={
            "pair_id": pair_id,
            "judgment": "",
        }, follow_redirects=True)
        api2 = client.get(f"/api/session/{session_id}")
        assert len(api2.get_json()["labels"]) == 0

    def test_missing_pair_id_rejected(self, client):
        client.post("/import", data={"json_text": _make_incidents(6)})
        resp = client.get("/")
        session_id = resp.data.decode().split("/session/")[1].split('"')[0]

        resp = client.post(f"/session/{session_id}/label", data={
            "judgment": "yes",
        }, follow_redirects=True)
        api = client.get(f"/api/session/{session_id}")
        assert len(api.get_json()["labels"]) == 0

    def test_long_notes_truncated(self, client):
        client.post("/import", data={"json_text": _make_incidents(6)})
        resp = client.get("/")
        session_id = resp.data.decode().split("/session/")[1].split('"')[0]
        api = client.get(f"/api/session/{session_id}")
        pair_id = api.get_json()["pairs"][0]["pair_id"]

        long_notes = "x" * 5000
        client.post(f"/session/{session_id}/label", data={
            "pair_id": pair_id,
            "judgment": "yes",
            "notes": long_notes,
        })
        api2 = client.get(f"/api/session/{session_id}")
        labels = api2.get_json()["labels"]
        assert len(labels) == 1
        assert len(labels[0]["notes"]) == 2000

    def test_long_labeler_id_truncated(self, client):
        client.post("/import", data={"json_text": _make_incidents(6)})
        resp = client.get("/")
        session_id = resp.data.decode().split("/session/")[1].split('"')[0]
        api = client.get(f"/api/session/{session_id}")
        pair_id = api.get_json()["pairs"][0]["pair_id"]

        client.post(f"/session/{session_id}/label", data={
            "pair_id": pair_id,
            "judgment": "no",
            "labeler_id": "a" * 1000,
        })
        api2 = client.get(f"/api/session/{session_id}")
        assert len(api2.get_json()["labels"][0]["labeler_id"]) == 255

    def test_duplicate_label_same_pair(self, client):
        """Submitting two labels for the same pair should store both (different reviewers)."""
        client.post("/import", data={"json_text": _make_incidents(6)})
        resp = client.get("/")
        session_id = resp.data.decode().split("/session/")[1].split('"')[0]
        api = client.get(f"/api/session/{session_id}")
        pair_id = api.get_json()["pairs"][0]["pair_id"]

        client.post(f"/session/{session_id}/label", data={
            "pair_id": pair_id, "judgment": "yes", "labeler_id": "alice",
        })
        client.post(f"/session/{session_id}/label", data={
            "pair_id": pair_id, "judgment": "no", "labeler_id": "bob",
        })
        api2 = client.get(f"/api/session/{session_id}")
        labels = api2.get_json()["labels"]
        assert len(labels) == 2

    def test_nonexistent_session_api(self, client):
        resp = client.get("/api/session/doesnotexist")
        assert resp.status_code == 404

    def test_nonexistent_session_export(self, client):
        resp = client.get("/api/export/doesnotexist")
        assert resp.status_code == 404

    def test_dashboard_with_zero_labels(self, client):
        client.post("/import", data={"json_text": _make_incidents(6)})
        resp = client.get("/")
        session_id = resp.data.decode().split("/session/")[1].split('"')[0]
        resp = client.get(f"/dashboard/{session_id}")
        assert resp.status_code == 200
        assert b"No labels yet" in resp.data or b"Start labeling" in resp.data
