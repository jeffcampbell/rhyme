#!/usr/bin/env python3
"""Classify real cloud incidents against our taxonomy to validate fit."""

import json
import os
import random
import urllib.request
from collections import Counter
from pathlib import Path

API_KEY = os.environ["ANTHROPIC_API_KEY"]

examples = json.loads(Path("/app/data/real_incidents/style_examples.json").read_text())
rng = random.Random(42)

# Pick 40 diverse examples with enough text
good = [e for e in examples if len(e["text"]) > 150]
rng.shuffle(good)
sample = good[:40]

taxonomy_classes = [
    "retry_storm", "downstream_slowdown", "dependency_outage", "dns_resolution_failure",
    "certificate_expiry", "connection_pool_exhaustion", "memory_leak", "traffic_allocation",
    "cpu_throttling", "disk_pressure", "code_regression", "config_regression",
    "network_partition", "load_balancer_misconfiguration", "traffic_spike",
    "cascading_timeout", "schema_migration_failure", "data_corruption",
    "clock_skew", "race_condition",
]

incidents_block = ""
for i, ex in enumerate(sample):
    incidents_block += f'[{i+1}] ({ex["provider"].upper()}) {ex["text"][:400]}\n\n'

prompt = f"""You are classifying real cloud provider incidents against a taxonomy of 20 proximal cause classes.

TAXONOMY:
{chr(10).join(f"- {c}" for c in taxonomy_classes)}

For each incident below, classify it into the BEST matching class. If none fit well, use "NONE" and explain why in the note field.

Return a JSON array: [{{"id": 1, "class": "...", "confidence": "high/medium/low", "note": "..."}}]

INCIDENTS:
{incidents_block}

Return ONLY the JSON array."""

body = json.dumps({
    "model": "claude-sonnet-4-6",
    "max_tokens": 4000,
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

text = data["content"][0]["text"].strip()
if "```" in text:
    text = text.split("```")[1]
    if text.startswith("json"):
        text = text[4:]
    text = text.split("```")[0]

results = json.loads(text.strip())

# Analyze
class_counts = Counter(r["class"] for r in results)
conf_counts = Counter(r["confidence"] for r in results)
none_count = sum(1 for r in results if r["class"] == "NONE")

print(f"Classified {len(results)} real incidents against 20-class taxonomy")
print(f"Confidence: {dict(conf_counts)}")
print(f"NONE (no fit): {none_count}/{len(results)} ({100*none_count//len(results)}%)")
print()
print("Class distribution:")
for cls, count in class_counts.most_common():
    print(f"  {cls:35s} {count}")
print()
print("Low-confidence and NONE classifications:")
for r in results:
    if r["confidence"] == "low" or r["class"] == "NONE":
        ex = sample[r["id"] - 1]
        print(f'  [{r["id"]}] {r["class"]:30s} ({r["confidence"]}) {ex["provider"].upper()}: {ex["text"][:120]}')
        if r.get("note"):
            print(f"       Note: {r['note']}")
