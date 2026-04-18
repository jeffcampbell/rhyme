"""
Subprocess adapter: language-agnostic model connector.

Spawns a child process and communicates via JSON-lines on stdin/stdout.
Any language can implement the protocol — just read a JSON line, write a JSON line.

Protocol:
  - Harness writes one JSON object per line to subprocess stdin:
    {"task": "retrieve", "query": <IncidentPayload>, "corpus": [<IncidentPayload>, ...], "k": 10}
    {"task": "remediate", "query": <IncidentPayload>, "top_matches": [<IncidentPayload>, ...], "choices": [<RemediationChoice>, ...]}

  - Subprocess writes one JSON object per line to stdout:
    For retrieve: {"ranked_matches": [{"incident_id": "...", "confidence": 0.9}, ...], "token_usage": {"input_tokens": N, "output_tokens": N}}
    For remediate: {"selected_label": "A", "token_usage": {...}}

  Token usage is optional in both directions.
  The subprocess stays alive for all queries (not restarted per query).
"""

from __future__ import annotations

import json
import subprocess

from .harness import Adapter, RetrieveOutput
from .models import (
    IncidentPayload,
    RankedMatch,
    RemediationChoice,
    TokenUsage,
)


class SubprocessAdapter(Adapter):
    """Adapter that delegates to an external process via JSON-lines on stdin/stdout."""

    def __init__(self, command: list[str], timeout: float = 300):
        """
        Args:
            command: Command to run as subprocess (e.g., ["python", "my_model.py"]).
            timeout: Seconds to wait for each response.
        """
        self._command = command
        self._timeout = timeout
        self._proc: subprocess.Popen | None = None

    def _ensure_started(self) -> subprocess.Popen:
        if self._proc is None or self._proc.poll() is not None:
            self._proc = subprocess.Popen(
                self._command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # line-buffered
            )
        return self._proc

    def _send_receive(self, request: dict) -> dict:
        proc = self._ensure_started()
        assert proc.stdin is not None
        assert proc.stdout is not None

        line = json.dumps(request, separators=(",", ":")) + "\n"
        proc.stdin.write(line)
        proc.stdin.flush()

        response_line = proc.stdout.readline()
        if not response_line:
            stderr = proc.stderr.read() if proc.stderr else ""
            raise RuntimeError(
                f"Subprocess exited without response. stderr: {stderr[:500]}"
            )
        return json.loads(response_line)

    def retrieve(
        self,
        query: IncidentPayload,
        corpus: list[IncidentPayload],
        k: int = 10,
    ) -> RetrieveOutput:
        request = {
            "task": "retrieve",
            "query": query.model_dump(),
            "corpus": [p.model_dump() for p in corpus],
            "k": k,
        }
        response = self._send_receive(request)

        matches = [
            RankedMatch(
                incident_id=m["incident_id"],
                confidence=m.get("confidence", 0.5),
                reasoning=m.get("reasoning"),
            )
            for m in response.get("ranked_matches", [])
        ]

        token_usage = None
        if "token_usage" in response and response["token_usage"]:
            tu = response["token_usage"]
            token_usage = TokenUsage(
                input_tokens=tu.get("input_tokens", 0),
                output_tokens=tu.get("output_tokens", 0),
            )

        return RetrieveOutput(matches=matches, token_usage=token_usage)

    def remediate(
        self,
        query: IncidentPayload,
        top_matches: list[IncidentPayload],
        choices: list[RemediationChoice],
    ) -> str | None:
        request = {
            "task": "remediate",
            "query": query.model_dump(),
            "top_matches": [p.model_dump() for p in top_matches],
            "choices": [c.model_dump() for c in choices],
        }
        response = self._send_receive(request)
        return response.get("selected_label")

    def close(self):
        if self._proc and self._proc.poll() is None:
            self._proc.stdin.close()
            self._proc.wait(timeout=10)

    def __del__(self):
        self.close()
