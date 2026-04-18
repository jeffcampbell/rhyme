# Sifter

**A benchmark for evaluating how well LLMs correlate microservice incidents by proximal cause — not just by symptoms.**

An "AI SRE" tool that surfaces similar past incidents has a high risk of producing red herrings: incidents that share symptoms but have completely different proximal causes. A tool that confidently surfaces a symptom-similar-but-cause-different incident can actively harm response (wrong remediation applied, real cause missed).

This benchmark measures whether a model can tell the difference.

## What it measures

Given a query incident, how well can a model:

1. **Retrieve** past incidents sharing the same *proximal cause* from a corpus of 1,000 synthetic incidents across 20 cause classes
2. **Re-rank** pre-retrieved candidates to separate cause-similar from symptom-similar incidents
3. **Recommend** a remediation that would actually fix the proximal cause (not just mask the symptom)

Scored separately by confusability tier (easy / medium / hard) so that strong performance on easy cases doesn't mask failure on the hard ones.

## Quick start

```sh
# Build
docker build -t sifter .

# Generate corpus + queries
docker run --rm -v $(pwd)/data:/app/data --entrypoint sifter-generate sifter \
  --output-dir /app/data --prose-pools-dir /app/data/prose_pools --incidents-per-class 50

# Run shipped baselines (all 3 tasks)
docker run --rm -v $(pwd)/data:/app/data --entrypoint sifter-run sifter \
  --corpus /app/data/corpus.json --queries /app/data/query_payloads.json \
  --output-dir /app/data/results

# Run with an LLM (just tasks 2+3 to save cost)
docker run --rm -v $(pwd)/data:/app/data \
  -e ANTHROPIC_API_KEY=sk-... -e SIFTER_MODEL=claude-haiku-4-5-20251001 \
  --entrypoint sifter-run sifter \
  --corpus /app/data/corpus.json --queries /app/data/query_payloads.json \
  --output-dir /app/data/results --tasks 2,3 \
  --adapter "python /app/examples/anthropic_adapter.py"

# Score
docker run --rm -v $(pwd)/data:/app/data --entrypoint sifter-score sifter \
  --corpus /app/data/corpus.json --queries /app/data/queries.json \
  --results /app/data/results/bm25_results.json
```

## Sample output

```
=== Sifter Score Report (k=10) ===

Per-tier results (primary — no composite score):
  easy    n=60  Ret@k=1.000  Prec@k=0.838  Lift=16.767  ECE=0.075
  medium  n=80  Ret@k=1.000  Prec@k=0.849  Lift=16.975  ECE=0.071
  hard    n=60  Ret@k=1.000  Prec@k=0.803  Lift=16.067  ECE=0.123

--- Confusion matrix (top wrong predictions) ---
  downstream_slowdown -> cascading_timeout          count=1
```

## Baseline reference scores (1,000 incidents, 200 queries)

| Baseline | Retrieval@10 | Precision@10 (easy) | (medium) | (hard) | ECE |
|----------|-------------|-------|--------|------|-----|
| Random | 0.340 | 0.038 | 0.039 | 0.035 | 0.461 |
| TF-IDF cosine | 0.955 | 0.563 | 0.588 | 0.502 | 0.059 |
| BM25 | 1.000 | 0.838 | 0.849 | 0.803 | 0.088 |

BM25 is the critical reference point — it shows what keyword matching alone can achieve.

### LLM results (Claude, 20 queries, 400 incidents)

**Task 2 — Reasoning-only (re-rank BM25 top-20):**

| Model | Precision@10 | Tokens/query | Prec/1k tokens |
|-------|-------------|-------------|----------------|
| BM25 | 0.790 | 0 | free |
| **Haiku** | **0.825** | 2,093 | 0.394 |
| **Sonnet** | **0.835** | 2,183 | 0.383 |

**Task 3 — Remediation (multiple-choice):**

| Model | Correct fix | Would worsen |
|-------|------------|-------------|
| Random | 21% | 16% |
| Haiku | 70% | 5% |
| **Sonnet** | **90%** | **0%** |

LLMs beat BM25 on reasoning-only re-ranking and dominate remediation. The most cost-effective approach: let BM25 do cheap retrieval, let the LLM reason on a small candidate set (30x fewer tokens, better results).

## Tasks and cost control

The benchmark has 3 tasks with very different token costs:

| Task | What it does | Tokens/query (1k incidents) | When to use |
|------|-------------|---------------------------|-------------|
| 1 | Full-corpus retrieval | ~75,000 | Evaluating retrieval capability |
| 2 | Re-rank BM25 top-20 | ~2,000 | Evaluating reasoning (recommended) |
| 3 | Multiple-choice remediation | ~500 | Evaluating domain knowledge |

Select tasks with `--tasks`:

```sh
sifter-run --tasks 2,3 --adapter "..."   # Skip expensive Task 1 (~$2 vs ~$50)
sifter-run --tasks 1,2,3 --adapter "..."  # Full evaluation
sifter-run --tasks 3 --adapter "..."      # Just remediation (~$0.10)
```

The CLI shows estimated token counts before any API calls are made.

## Plug in your model

### Option A: Any language (subprocess protocol)

Write a program that reads JSON from stdin and writes JSON to stdout:

```python
#!/usr/bin/env python3
import json, sys

for line in sys.stdin:
    req = json.loads(line)
    corpus = req["corpus"]
    k = req["k"]

    # Your model logic here
    matches = [{"incident_id": c["incident_id"], "confidence": 0.5} for c in corpus[:k]]

    sys.stdout.write(json.dumps({"ranked_matches": matches}) + "\n")
    sys.stdout.flush()
```

```sh
docker run --rm -v $(pwd)/data:/app/data \
  -e ANTHROPIC_API_KEY=sk-... -e SIFTER_MODEL=claude-haiku-4-5-20251001 \
  --entrypoint sifter-run sifter \
  --corpus /app/data/corpus.json --queries /app/data/query_payloads.json \
  --output-dir /app/data/results --adapter "python /app/examples/anthropic_adapter.py"
```

### Pre-built adapters

| Adapter | Providers | Auth env var |
|---------|-----------|-------------|
| [`anthropic_adapter.py`](examples/anthropic_adapter.py) | Anthropic (Claude) | `ANTHROPIC_API_KEY` |
| [`openai_compat_adapter.py`](examples/openai_compat_adapter.py) | OpenAI, Mistral, Groq, Together, Fireworks, DeepSeek, Perplexity, OpenRouter, Ollama, vLLM, llama.cpp | `OPENAI_API_KEY` + `OPENAI_BASE_URL` |
| [`gemini_adapter.py`](examples/gemini_adapter.py) | Google Gemini (AI Studio) | `GEMINI_API_KEY` |
| [`bedrock_adapter.py`](examples/bedrock_adapter.py) | AWS Bedrock (all models) | AWS credentials + `AWS_REGION` |

All adapters use `SIFTER_MODEL` env var to select the model.

### Option B: Python

```python
from sifter_bench.harness import Adapter
from sifter_bench.models import IncidentPayload, RankedMatch

class MyModel(Adapter):
    def retrieve(self, query: IncidentPayload, corpus: list[IncidentPayload], k: int = 10):
        # Your logic here
        return [RankedMatch(incident_id=c.incident_id, confidence=0.5) for c in corpus[:k]]
```

### Efficiency comparison

Report `token_usage` in your adapter's response to compare prompting strategies on a quality-vs-cost frontier:

```json
{"ranked_matches": [...], "token_usage": {"input_tokens": 5000, "output_tokens": 200}}
```

The scorer will report tokens/query, tokens/correct-match, and precision-per-1k-tokens.

## Evaluate against your own incidents

A web-based labeling tool lets you evaluate model correlation quality against your organization's real incident data.

```sh
# Production (PostgreSQL + Gunicorn)
docker compose up -d
# visit http://localhost:8080

# Local dev (SQLite, no PostgreSQL needed)
docker compose up web-dev
```

Then visit `http://localhost:8080`:

1. **Import** your incidents as JSON (just `id`, `summary`, `timestamp` — no logs or alerts needed)
2. **Label** pairs side-by-side: "Would knowing about Incident A help you respond to Incident B?"
3. **Review** model performance vs. human labels on the dashboard

The tool highlights **red herrings** — cases where the model is confident but your SREs disagree.

### Deploying to AWS (or anywhere with Docker)

The `docker-compose.yml` runs on any Docker host. For AWS:

```sh
# On an EC2 instance or ECS with docker-compose:
SECRET_KEY=$(openssl rand -hex 32) docker compose up -d
```

For a managed setup: push the image to ECR, run on Fargate, point `DATABASE_URL` at an RDS PostgreSQL instance, and put an ALB in front using the `/health` endpoint.

## Taxonomy (20 cause classes)

| Family | Cause Classes |
|--------|--------------|
| Dependency | retry_storm, downstream_slowdown, dependency_outage, dns_resolution_failure, certificate_expiry, connection_pool_exhaustion |
| Resource | memory_leak, traffic_allocation, cpu_throttling, disk_pressure |
| Deploy | code_regression, config_regression |
| Network | network_partition, load_balancer_misconfiguration |
| Traffic | traffic_spike |
| Amplification | cascading_timeout |
| Data/Schema | schema_migration_failure, data_corruption |
| Timing | clock_skew, race_condition |

Each class has defined confusable partners (the "hard" pairs that the benchmark exists to test).

## Design

The full design rationale is in [`sifter-v1-spec.md`](sifter-v1-spec.md), including:

- Why synthetic data and the circularity problem (models never see labels or fingerprints)
- 7 failure modes pressure-tested during design
- The adversarial style probe that gates corpus quality
- Why there's no composite headline score

## What it explicitly does NOT measure

- Production readiness or performance on real incident data (use the web labeling tool for that)
- Performance on domains other than cloud-native Kubernetes HTTP microservices
- Agentic or multi-turn investigation
- Anything the cause-class taxonomy doesn't represent

## License

MIT
