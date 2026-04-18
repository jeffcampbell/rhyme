#!/usr/bin/env python3
"""
Google Gemini adapter for IncidentCorr-Bench.

Works with Google AI Studio (API key auth). For Vertex AI, use the
OpenAI-compatible endpoint instead (Vertex now supports it).

Usage:
  export GEMINI_API_KEY=... SIFTER_MODEL=gemini-2.0-flash
  sifter-run --adapter "python examples/gemini_adapter.py" ...
"""

import json
import os
import sys
import urllib.request

import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__))); from parse_utils import extract_json_array, extract_letter, normalize_matches

API_KEY = os.environ.get("GEMINI_API_KEY", "")
MODEL = os.environ.get("SIFTER_MODEL", "gemini-2.0-flash")


def call_gemini(prompt: str, max_tokens: int = 2000) -> tuple[str, dict]:
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": 0,
        },
    }).encode()

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}"
        f":generateContent?key={API_KEY}"
    )
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})

    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())

    text = data["candidates"][0]["content"]["parts"][0]["text"]
    usage_meta = data.get("usageMetadata", {})
    return text, {
        "input_tokens": usage_meta.get("promptTokenCount", 0),
        "output_tokens": usage_meta.get("candidatesTokenCount", 0),
    }


def handle_retrieve(request: dict) -> dict:
    corpus = request["corpus"]
    query = request["query"]
    k = request.get("k", 10)

    corpus_text = "\n".join(
        f"[{inc['incident_id']}] {inc['summary'][:300]}" for inc in corpus
    )

    prompt = f"""You are an expert SRE. Find the {k} most similar incidents by PROXIMAL CAUSE (not just symptoms).

QUERY INCIDENT:
{query['summary']}

CORPUS:
{corpus_text}

Return a JSON array: [{{"incident_id": "INC-...", "confidence": 0.95}}, ...]
Only the JSON array, nothing else."""

    try:
        text, usage = call_gemini(prompt)
        matches = normalize_matches(extract_json_array(text), k)
        return {"ranked_matches": matches, "token_usage": usage}
    except Exception as e:
        print(f"Retrieve error: {e}", file=sys.stderr)
        return {"ranked_matches": [], "token_usage": {}}


def handle_remediate(request: dict) -> dict:
    query = request["query"]
    choices = request.get("choices", [])
    choices_text = "\n".join(f"{c['label']}) {c['text']}" for c in choices)

    prompt = f"""You are an expert SRE. Which remediation best fixes the PROXIMAL CAUSE?

INCIDENT:
{query['summary']}

OPTIONS:
{choices_text}

Reply with ONLY the letter (A-E)."""

    try:
        text, usage = call_gemini(prompt, max_tokens=50)
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
