#!/usr/bin/env python3
"""
Anthropic Claude adapter for IncidentCorr-Bench.

Uses the Anthropic API directly. Supports Claude Haiku, Sonnet, and Opus.

Usage:
  export ANTHROPIC_API_KEY=sk-...
  export SIFTER_MODEL=claude-haiku-4-5-20251001  # or claude-sonnet-4-6
  sifter-run --adapter "python examples/anthropic_adapter.py" ...
"""

import json
import os
import sys
import urllib.request

import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__))); from parse_utils import extract_json_array, extract_letter, normalize_matches

API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL = os.environ.get("SIFTER_MODEL", "claude-haiku-4-5-20251001")


def call_claude(prompt: str, max_tokens: int = 2000) -> tuple[str, dict]:
    """Call the Anthropic API and return (response_text, usage_dict)."""
    body = json.dumps({
        "model": MODEL,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={
            "Content-Type": "application/json",
            "x-api-key": API_KEY,
            "anthropic-version": "2023-06-01",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())

    text = data["content"][0]["text"]
    usage = data.get("usage", {})
    return text, {
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
    }


def handle_retrieve(request: dict) -> dict:
    corpus = request["corpus"]
    query = request["query"]
    k = request.get("k", 10)

    # Build compact corpus index (summary only, to limit tokens)
    corpus_entries = []
    for inc in corpus:
        corpus_entries.append(f"[{inc['incident_id']}] {inc['summary'][:300]}")
    corpus_text = "\n".join(corpus_entries)

    prompt = f"""You are an expert SRE. Given a query incident, find the {k} most similar incidents from the corpus based on PROXIMAL CAUSE similarity (not just symptom overlap).

Two incidents share a proximal cause if the same fix would apply to both. Be careful of red herrings — incidents can look similar (same error codes, same affected services) but have completely different underlying causes.

QUERY INCIDENT:
{query['summary']}

CORPUS:
{corpus_text}

Return a JSON array of the top {k} matches ranked by proximal-cause similarity. Use this exact format:
[{{"incident_id": "INC-...", "confidence": 0.95}}, ...]

confidence: 0.0 = definitely different proximal cause, 1.0 = definitely same proximal cause.
Only return the JSON array, nothing else."""

    try:
        text, usage = call_claude(prompt)
        matches = normalize_matches(extract_json_array(text), k)
        return {"ranked_matches": matches, "token_usage": usage}
    except Exception as e:
        print(f"Retrieve error: {e}", file=sys.stderr)
        return {"ranked_matches": [], "token_usage": {}}


def handle_remediate(request: dict) -> dict:
    query = request["query"]
    choices = request.get("choices", [])

    choices_text = "\n".join(f"{c['label']}) {c['text']}" for c in choices)

    prompt = f"""You are an expert SRE. Given this incident, which remediation would best fix the PROXIMAL CAUSE (not just mask symptoms)?

INCIDENT:
{query['summary']}

OPTIONS:
{choices_text}

Think carefully about what the proximal cause is, then reply with ONLY the letter (A, B, C, D, or E)."""

    try:
        text, usage = call_claude(prompt, max_tokens=50)
        return {"selected_label": extract_letter(text), "token_usage": usage}
    except Exception as e:
        print(f"Remediate error: {e}", file=sys.stderr)
        return {"selected_label": "A"}


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        request = json.loads(line)
        task = request.get("task", "retrieve")
        if task == "retrieve":
            response = handle_retrieve(request)
        elif task == "remediate":
            response = handle_remediate(request)
        else:
            response = {"error": f"Unknown task: {task}"}
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
