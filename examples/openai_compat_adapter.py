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
  export OPENAI_API_KEY=sk-... RHYME_MODEL=gpt-4o-mini
  rhyme-run --adapter "python examples/openai_compat_adapter.py" ...

  # Ollama (local)
  export OPENAI_BASE_URL=http://localhost:11434/v1 OPENAI_API_KEY=unused RHYME_MODEL=llama3
  rhyme-run --adapter "python examples/openai_compat_adapter.py" ...

  # Together AI
  export OPENAI_BASE_URL=https://api.together.xyz/v1 OPENAI_API_KEY=... RHYME_MODEL=meta-llama/Llama-3-70b-chat-hf
  rhyme-run --adapter "python examples/openai_compat_adapter.py" ...

  # OpenRouter, reasoning off + pinned provider (reproducible, matches a local
  # reasoning-off config so hosted iteration transfers back to a local run):
  export OPENAI_BASE_URL=https://openrouter.ai/api/v1 OPENAI_API_KEY=sk-or-...
  export RHYME_MODEL=some/model RHYME_EXTRA_BODY='{"reasoning":{"enabled":false}}'
  export RHYME_PROVIDER=DeepInfra RHYME_MODEL_LOG=/app/data/served_models.jsonl
  rhyme-run --tasks 3 --adapter "python examples/openai_compat_adapter.py" ...

Extra knobs:
  RHYME_EXTRA_BODY  JSON object shallow-merged into every request body; its keys
                    win over everything else (provider pin, defaults). Aborts at
                    startup if not valid JSON, so a config field is never silently
                    dropped.
  RHYME_PROVIDER    Comma-separated OpenRouter provider preference order; builds
                    the `provider` routing block with fallbacks off (set
                    RHYME_PROVIDER_ALLOW_FALLBACKS=true to allow them).
"""

import json
import os
import sys
import urllib.request

import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__))); from parse_utils import extract_json_array, extract_letter, normalize_matches

API_KEY = os.environ.get("OPENAI_API_KEY", "")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL = os.environ.get("RHYME_MODEL", "gpt-4o-mini")
# Optional: append the model actually served per call to this JSONL file. Used
# for analysing routers (e.g. openrouter/auto-beta) where the served model
# differs per request. Not part of the scored output.
MODEL_LOG = os.environ.get("RHYME_MODEL_LOG", "")

# Output token budgets, overridable via env. Defaults are generous enough for
# reasoning models (which spend hidden tokens before answering) so their JSON
# array / letter answer isn't truncated away.
RETRIEVE_MAX_TOKENS = int(os.environ.get("RHYME_RETRIEVE_MAX_TOKENS", "6000"))
REMEDIATE_MAX_TOKENS = int(os.environ.get("RHYME_REMEDIATE_MAX_TOKENS", "2000"))

# Extra request-body fields merged into every /chat/completions call. A JSON
# object string; its top-level keys are shallow-merged into the body and win
# over everything set here (including the conveniences below). Use it to pass
# provider-specific fields that aren't part of the OpenAI schema, e.g.:
#   OpenRouter reasoning off:  RHYME_EXTRA_BODY='{"reasoning":{"enabled":false}}'
#   vLLM/LM Studio thinking off: RHYME_EXTRA_BODY='{"reasoning_effort":"none"}'
# A malformed value aborts at startup rather than silently sending the request
# without the field — a silent drop would quietly change what config you tested.
try:
    EXTRA_BODY = json.loads(os.environ.get("RHYME_EXTRA_BODY", "") or "{}")
except json.JSONDecodeError as e:
    print(f"RHYME_EXTRA_BODY is not valid JSON: {e}", file=sys.stderr)
    sys.exit(1)
if not isinstance(EXTRA_BODY, dict):
    print("RHYME_EXTRA_BODY must be a JSON object", file=sys.stderr)
    sys.exit(1)

# OpenRouter provider pin. RHYME_PROVIDER is a comma-separated preference order
# (e.g. "DeepInfra,Together"); it builds the OpenRouter `provider` routing block
# so you don't silently get a different backend (and quantization) per request.
# Fallbacks are disabled by default for reproducibility — set
# RHYME_PROVIDER_ALLOW_FALLBACKS=true to allow routing outside the pinned list.
PROVIDERS = [p.strip() for p in os.environ.get("RHYME_PROVIDER", "").split(",") if p.strip()]
ALLOW_FALLBACKS = os.environ.get("RHYME_PROVIDER_ALLOW_FALLBACKS", "false").lower() in (
    "1", "true", "yes",
)


def call_llm(prompt: str, max_tokens: int = 2000, system: str | None = None) -> tuple[str, dict, str]:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    # Provider pin (convenience) first, then RHYME_EXTRA_BODY wins over everything.
    if PROVIDERS:
        payload["provider"] = {"order": PROVIDERS, "allow_fallbacks": ALLOW_FALLBACKS}
    payload.update(EXTRA_BODY)
    body = json.dumps(payload).encode()

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
    # OpenRouter (and the auto router) echoes the model that actually served
    # the request in the top-level "model" field; fall back to what we asked for.
    resolved_model = data.get("model", MODEL)
    return text, {
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
    }, resolved_model


def log_resolved_model(task: str, resolved_model: str, request: dict, usage: dict) -> None:
    if not MODEL_LOG:
        return
    record = {
        "task": task,
        "requested_model": MODEL,
        "resolved_model": resolved_model,
        "query_id": (request.get("query") or {}).get("incident_id"),
        "token_usage": usage,
    }
    try:
        with open(MODEL_LOG, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        print(f"model-log error: {e}", file=sys.stderr)


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
        text, usage, resolved_model = call_llm(
            prompt, max_tokens=RETRIEVE_MAX_TOKENS, system=request.get("system_prompt"),
        )
        log_resolved_model("retrieve", resolved_model, request, usage)
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
        text, usage, resolved_model = call_llm(
            prompt, max_tokens=REMEDIATE_MAX_TOKENS, system=request.get("system_prompt"),
        )
        log_resolved_model("remediate", resolved_model, request, usage)
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
