#!/usr/bin/env python3
"""
AWS Bedrock adapter for IncidentCorr-Bench.

Uses the Bedrock Converse API for a unified interface across all Bedrock
models (Claude, Llama, Mistral, Titan, etc.).

Requires: boto3, AWS credentials configured (env vars, ~/.aws/credentials, or IAM role).

Usage:
  export AWS_REGION=us-east-1 SIFTER_MODEL=anthropic.claude-3-haiku-20240307-v1:0
  sifter-run --adapter "python examples/bedrock_adapter.py" ...

  # Or with explicit credentials:
  export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... AWS_REGION=us-east-1
  sifter-run --adapter "python examples/bedrock_adapter.py" ...
"""

import json
import os
import sys

import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__))); from parse_utils import extract_json_array, extract_letter, normalize_matches

MODEL = os.environ.get("SIFTER_MODEL", "anthropic.claude-3-haiku-20240307-v1:0")
REGION = os.environ.get("AWS_REGION", "us-east-1")

# Lazy-loaded boto3 client
_client = None


def _get_client():
    global _client
    if _client is None:
        import boto3
        _client = boto3.client("bedrock-runtime", region_name=REGION)
    return _client


def call_bedrock(prompt: str, max_tokens: int = 2000) -> tuple[str, dict]:
    client = _get_client()

    response = client.converse(
        modelId=MODEL,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": max_tokens, "temperature": 0},
    )

    text = response["output"]["message"]["content"][0]["text"]
    usage = response.get("usage", {})
    return text, {
        "input_tokens": usage.get("inputTokens", 0),
        "output_tokens": usage.get("outputTokens", 0),
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
        text, usage = call_bedrock(prompt)
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
        text, usage = call_bedrock(prompt, max_tokens=50)
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
