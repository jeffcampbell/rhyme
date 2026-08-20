"""Tests for the subprocess adapter."""

import json
import sys
import textwrap
from pathlib import Path

import pytest

from rhyme_bench.subprocess_adapter import SubprocessAdapter
from rhyme_bench.models import IncidentPayload, RankedMatch, TokenUsage
from rhyme_bench.harness import RetrieveOutput


@pytest.fixture
def echo_script(tmp_path):
    """Create a minimal subprocess adapter script for testing."""
    script = tmp_path / "test_adapter.py"
    script.write_text(textwrap.dedent("""\
        import json, sys
        for line in sys.stdin:
            req = json.loads(line.strip())
            if req.get("task") == "retrieve":
                corpus = req.get("corpus", [])
                k = req.get("k", 10)
                matches = [{"incident_id": c["incident_id"], "confidence": 0.5} for c in corpus[:k]]
                resp = {"ranked_matches": matches, "token_usage": {"input_tokens": 100, "output_tokens": 50}}
            elif req.get("task") == "remediate":
                resp = {"selected_label": "A"}
            else:
                resp = {"error": "unknown"}
            sys.stdout.write(json.dumps(resp) + "\\n")
            sys.stdout.flush()
    """))
    return str(script)


def test_subprocess_retrieve(echo_script, small_corpus):
    adapter = SubprocessAdapter([sys.executable, echo_script])
    payloads = small_corpus.payloads()
    result = adapter.retrieve(payloads[0], payloads[1:5], k=3)

    assert isinstance(result, RetrieveOutput)
    assert len(result.matches) <= 3
    assert result.token_usage is not None
    assert result.token_usage.input_tokens == 100
    adapter.close()


def test_subprocess_remediate(echo_script, small_corpus):
    adapter = SubprocessAdapter([sys.executable, echo_script])
    from rhyme_bench.models import RemediationChoice
    choices = [
        RemediationChoice(label="A", text="Fix", grade="fixed"),
        RemediationChoice(label="B", text="Mask", grade="masks_symptom"),
    ]
    label = adapter.remediate(small_corpus.payloads()[0], [], choices)
    assert label == "A"
    adapter.close()


def test_subprocess_stays_alive(echo_script, small_corpus):
    """Subprocess should handle multiple queries without restarting."""
    adapter = SubprocessAdapter([sys.executable, echo_script])
    payloads = small_corpus.payloads()

    for i in range(5):
        result = adapter.retrieve(payloads[i], payloads[:3], k=2)
        assert len(result.matches) <= 2

    adapter.close()


@pytest.fixture
def bad_script(tmp_path):
    """Script that exits immediately."""
    script = tmp_path / "bad_adapter.py"
    script.write_text("import sys; sys.exit(1)")
    return str(script)


def test_subprocess_handles_crash(bad_script, small_corpus):
    adapter = SubprocessAdapter([sys.executable, bad_script])
    with pytest.raises(RuntimeError, match="Subprocess exited"):
        adapter.retrieve(small_corpus.payloads()[0], small_corpus.payloads()[:3], k=2)
    adapter.close()


@pytest.fixture
def spy_script(tmp_path):
    """Adapter that echoes the `system_prompt` it received back to the caller.

    Retrieve puts the seen value in each match's `reasoning`; remediate returns
    it as the `selected_label`. Absence of the field is reported as the sentinel
    "<MISSING>" so tests can distinguish an omitted field from an empty string.
    """
    script = tmp_path / "spy_adapter.py"
    script.write_text(textwrap.dedent("""\
        import json, sys
        SENTINEL = "<MISSING>"
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            req = json.loads(line)
            seen = req.get("system_prompt", SENTINEL)
            if req.get("task") == "retrieve":
                corpus = req.get("corpus", [])
                k = req.get("k", 10)
                matches = [
                    {"incident_id": c["incident_id"], "confidence": 0.5, "reasoning": seen}
                    for c in corpus[:k]
                ]
                resp = {"ranked_matches": matches}
            elif req.get("task") == "remediate":
                resp = {"selected_label": seen}
            else:
                resp = {"error": "unknown"}
            sys.stdout.write(json.dumps(resp) + "\\n")
            sys.stdout.flush()
    """))
    return str(script)


def _remediate(adapter, payload):
    from rhyme_bench.models import RemediationChoice
    return adapter.remediate(
        payload, [], [RemediationChoice(label="A", text="Fix", grade="fixed")]
    )


def test_system_prompt_per_task_distinct(spy_script, small_corpus):
    """Per-task prompts thread through independently to each task."""
    adapter = SubprocessAdapter(
        [sys.executable, spy_script],
        system_prompt_retrieve="RET_SYS",
        system_prompt_remediate="REM_SYS",
    )
    payloads = small_corpus.payloads()

    result = adapter.retrieve(payloads[0], payloads[1:4], k=2)
    assert result.matches[0].reasoning == "RET_SYS"

    assert _remediate(adapter, payloads[0]) == "REM_SYS"
    adapter.close()


def test_system_prompt_omitted_when_unset(spy_script, small_corpus):
    """With no system prompt configured, the field is omitted entirely.

    This guarantees existing runs/results are unchanged by the feature.
    """
    adapter = SubprocessAdapter([sys.executable, spy_script])
    payloads = small_corpus.payloads()

    result = adapter.retrieve(payloads[0], payloads[1:4], k=2)
    assert result.matches[0].reasoning == "<MISSING>"

    assert _remediate(adapter, payloads[0]) == "<MISSING>"
    adapter.close()


def test_system_prompt_retrieve_only(spy_script, small_corpus):
    """A retrieve-only prompt does not leak into the remediate task."""
    adapter = SubprocessAdapter(
        [sys.executable, spy_script],
        system_prompt_retrieve="ONLY_RET",
    )
    payloads = small_corpus.payloads()

    result = adapter.retrieve(payloads[0], payloads[1:4], k=2)
    assert result.matches[0].reasoning == "ONLY_RET"

    assert _remediate(adapter, payloads[0]) == "<MISSING>"
    adapter.close()


def test_system_prompt_empty_string_treated_as_unset(spy_script, small_corpus):
    """An empty-string prompt is falsy and omitted, not sent as ''."""
    adapter = SubprocessAdapter(
        [sys.executable, spy_script],
        system_prompt_retrieve="",
    )
    payloads = small_corpus.payloads()
    result = adapter.retrieve(payloads[0], payloads[1:4], k=2)
    assert result.matches[0].reasoning == "<MISSING>"
    adapter.close()
