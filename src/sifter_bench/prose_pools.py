"""
LLM-generated prose pools for corpus generation.

Each cause class has a pool of varied prose: summaries, alert messages,
and log lines. The generator samples from these pools instead of using
a small set of hardcoded templates, breaking the lexical regularity
that lets keyword-based retrievers cheat (spec §10 FM1).

Pools are stored as JSON files in data/prose_pools/{cause_class}.json.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from .taxonomy import CauseClass


class AlertPool(BaseModel):
    severity: str
    messages: list[str]


class LogPool(BaseModel):
    level: str
    messages: list[str]


class ProsePool(BaseModel):
    """Expanded prose pool for a single cause class."""

    cause_class: str
    summaries: list[str]
    alerts: list[AlertPool]
    logs: list[LogPool]


def load_prose_pool(cause_class: CauseClass, pools_dir: Path) -> ProsePool | None:
    """Load a prose pool from disk, or return None if it doesn't exist."""
    path = pools_dir / f"{cause_class.value}.json"
    if not path.exists():
        return None
    return ProsePool.model_validate_json(path.read_text())


def save_prose_pool(pool: ProsePool, pools_dir: Path) -> None:
    """Save a prose pool to disk."""
    pools_dir.mkdir(parents=True, exist_ok=True)
    path = pools_dir / f"{pool.cause_class}.json"
    path.write_text(pool.model_dump_json(indent=2))


def load_all_prose_pools(pools_dir: Path) -> dict[CauseClass, ProsePool]:
    """Load all available prose pools."""
    pools: dict[CauseClass, ProsePool] = {}
    for cause_class in CauseClass:
        pool = load_prose_pool(cause_class, pools_dir)
        if pool is not None:
            pools[cause_class] = pool
    return pools


def build_agent_prompt(cause_class: CauseClass) -> str:
    """Build the prompt for a Sonnet agent to generate a prose pool.

    Imports taxonomy info at call time to build a complete prompt.
    """
    from .taxonomy import TAXONOMY, CAUSE_FAMILY_MAP, REMEDIATIONS
    from .archetypes import ARCHETYPES

    info = TAXONOMY[cause_class]
    archetype = ARCHETYPES[cause_class]
    remediation = REMEDIATIONS[cause_class]

    # Collect existing templates as examples
    existing_summaries = archetype.summary.templates
    existing_alerts = []
    for at in archetype.alerts:
        for msg in at.message_templates:
            existing_alerts.append(f"[{at.severity}] {msg}")
    existing_logs = []
    for lt in archetype.logs:
        for msg in lt.message_templates:
            existing_logs.append(f"[{lt.level}] {msg}")

    prompt = f"""You are generating diverse prose for a synthetic incident benchmark. Your job is to create realistic, varied incident descriptions for the cause class "{cause_class.value}".

## Cause class: {cause_class.value}
Family: {info.family.value}
Description: {info.description}

Distinguishing signals:
{chr(10).join(f"- {s}" for s in info.distinguishing_signals)}

Remediation: {remediation.canonical}

## What you must produce

Output a single JSON object with this exact schema:
```json
{{
  "cause_class": "{cause_class.value}",
  "summaries": ["...", "..."],
  "alerts": [
    {{"severity": "critical", "messages": ["...", "..."]}},
    {{"severity": "warning", "messages": ["...", "..."]}},
    {{"severity": "info", "messages": ["...", "..."]}}
  ],
  "logs": [
    {{"level": "ERROR", "messages": ["...", "..."]}},
    {{"level": "WARN", "messages": ["...", "..."]}},
    {{"level": "INFO", "messages": ["...", "..."]}}
  ]
}}
```

## Requirements

### Summaries (generate exactly 20)
- Each summary is 2-5 sentences, written from a responder's perspective
- Vary sentence structure, vocabulary, and level of technical detail
- Some should be terse ("Service X went down. Cause: Y."), others more narrative
- Use template variables: {{service}}, {{downstream}}, {{db}}, {{error_pct}}, {{latency}}, {{rps}}, {{hours}}, {{deploy_id}}, {{deploy_time}}, {{error_type}}, {{object_type}}, {{mem_limit}}, {{start_heap}}, {{capacity}}, {{traffic_mult}}, {{amp_ratio}}, {{timeout_pct}}, {{p99}}, {{baseline}}, {{config_key}}, {{new_value}}, {{config_time}}, {{affected_count}}, {{endpoint}}
- CRITICAL: Do not use words that directly name the cause class. For example, if the cause is "retry_storm", do not use the phrase "retry storm" in every summary. Describe the symptoms and investigation, not the label.
- Each summary should feel like it was written by a different on-call engineer

### Alert messages (generate 10 critical, 10 warning, 5 info)
- Use the same template variables as summaries
- Vary the alert phrasing — real monitoring systems (Prometheus, Datadog, PagerDuty, OpsGenie) have different styles
- Some should be terse metric alerts, others should be more descriptive
- Include realistic metric values as template variables

### Log lines (generate 15 ERROR, 12 WARN, 8 INFO)
- Use the same template variables
- Vary between different programming languages/frameworks (Java, Go, Python, Node.js)
- Include realistic stack trace fragments, error messages, metric dumps
- Some should be from the origin service, others from downstream/infrastructure

## Examples (for style reference only — do NOT copy these)

Existing summaries:
{chr(10).join(f"- {s}" for s in existing_summaries[:2])}

Existing alerts:
{chr(10).join(f"- {a}" for a in existing_alerts[:4])}

Existing logs:
{chr(10).join(f"- {l}" for l in existing_logs[:4])}

## Style variation rules
- Vary sentence length: mix short (5-10 words) and long (20-40 words)
- Vary formality: some clinical, some conversational
- Vary specificity: some reference exact metrics, others describe trends
- Use different verbs for the same concept (e.g., "spiked", "climbed", "jumped", "surged", "rose")
- Reference different monitoring tools (Grafana, Datadog, kubectl, CloudWatch)
- Reference different infrastructure (AWS, GCP, bare metal, different K8s flavors)

Output ONLY the JSON object, no other text."""

    return prompt
