#!/usr/bin/env python3
"""
Example subprocess adapter: returns the first k corpus incidents as matches.

This is a minimal reference implementation of the subprocess protocol.
Replace the logic in handle_retrieve() with your model's actual retrieval.

Usage:
  rhyme-run --corpus data/corpus.json --queries data/query_payloads.json \
          --adapter "python examples/echo_adapter.py"
"""

import json
import sys


def handle_retrieve(request: dict) -> dict:
    """Replace this with your model's retrieval logic."""
    corpus = request["corpus"]
    k = request.get("k", 10)

    # Naive: return first k corpus incidents with declining confidence
    ranked = [
        {
            "incident_id": inc["incident_id"],
            "confidence": round(1.0 - i * 0.05, 3),
        }
        for i, inc in enumerate(corpus[:k])
    ]

    return {
        "ranked_matches": ranked,
        # Optional: report token usage for efficiency metrics
        # "token_usage": {"input_tokens": 1000, "output_tokens": 200},
    }


def handle_remediate(request: dict) -> dict:
    """Replace this with your model's remediation logic."""
    choices = request.get("choices", [])
    # Naive: pick the first choice
    label = choices[0]["label"] if choices else "A"
    return {"selected_label": label}


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
