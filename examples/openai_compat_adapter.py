#!/usr/bin/env python3
"""
OpenAI-compatible adapter — works with any provider that implements
the /v1/chat/completions endpoint.

Tested with: OpenAI, Mistral, Groq, Together AI, Fireworks AI, DeepSeek,
Perplexity, OpenRouter, Anyscale, Ollama, vLLM, llama.cpp, LocalAI.

For Azure OpenAI, set OPENAI_BASE_URL to your deployment endpoint and
OPENAI_API_KEY to your API key — Azure's OpenAI endpoint is compatible.

Usage:
  # OpenAI
  export OPENAI_API_KEY=sk-... SIFTER_MODEL=gpt-4o-mini
  sifter-run --adapter "python examples/openai_compat_adapter.py" ...

  # Ollama (local)
  export OPENAI_BASE_URL=http://localhost:11434/v1 OPENAI_API_KEY=unused SIFTER_MODEL=llama3
  sifter-run --adapter "python examples/openai_compat_adapter.py" ...

  # Together AI
  export OPENAI_BASE_URL=https://api.together.xyz/v1 OPENAI_API_KEY=... SIFTER_MODEL=meta-llama/Llama-3-70b-chat-hf
  sifter-run --adapter "python examples/openai_compat_adapter.py" ...
"""

import json
import os
import sys
import urllib.request

import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__))); from parse_utils import extract_json_array, extract_letter, normalize_matches

API_KEY = os.environ.get("OPENAI_API_KEY", "")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL = os.environ.get("SIFTER_MODEL", "gpt-4o-mini")


def call_llm(prompt: str, max_tokens: int = 2000) -> tuple[str, dict]:
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }).encode()

    req = urllib.request.Request(
        f"{BASE_URL}/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())

    text = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    return text, {
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
    }


def handle_retrieve(request: dict) -> dict:
    corpus = request["corpus"]
    query = request["query"]
    k = request.get("k", 10)

    corpus_text = "\n".join(
        f"[{inc['incident_id']}] {inc['summary'][:300]}"
        for inc in corpus
    )

    prompt = f"""You are an expert SRE. Given a query incident, find the {k} most similar incidents from the corpus based on PROXIMAL CAUSE similarity (not just symptom overlap).

Two incidents share a proximal cause if the same fix would apply to both. Be careful of red herrings.

QUERY INCIDENT:
{query['summary']}

CORPUS:
{corpus_text}

Return a JSON array of the top {k} matches ranked by proximal-cause similarity:
[{{"incident_id": "INC-...", "confidence": 0.95}}, ...]

Only return the JSON array, nothing else."""

    try:
        text, usage = call_llm(prompt)
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
        text, usage = call_llm(prompt, max_tokens=50)
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
