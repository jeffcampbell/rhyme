# Sifter

**A benchmark for evaluating how well LLMs correlate microservice incidents by proximal cause — not just by symptoms.**

When an "AI SRE" tool surfaces a similar past incident, is it actually helpful — or is it a red herring? Incidents often share symptoms (same error codes, same affected services) while having completely different underlying causes. Surfacing the wrong one sends the on-call engineer down the wrong path.

Sifter measures whether a model can tell the difference.

## Sifter Scores

| Model | Precision@10 | ECE | Tokens/query | Correct fix | Would worsen |
|-------|-------------|-----|-------------|------------|-------------|
| **GPT-4.1 nano** | **0.818** | **0.037** | 1,875 | 90% | 0% |
| GPT-4o | 0.811 | 0.105 | 1,879 | 95% | 0% |
| Gemini 2.5 Flash | 0.805 | 0.102 | 2,307 | 94% | 0% |
| DeepSeek V3 | 0.804 | 0.115 | 1,933 | 72% | 4% |
| GPT-4o mini | 0.802 | 0.082 | 1,879 | 95% | 1% |
| Claude Sonnet 4 | 0.802 | 0.091 | 2,186 | 94% | 0% |
| Gemini 2.0 Flash | 0.789 | 0.132 | 2,243 | 89% | 0% |
| Claude Haiku 3.5 | 0.784 | 0.106 | 2,186 | 94% | 0% |
| Claude Opus 4 | 0.782 | 0.119 | 2,163 | **96%** | 0% |
| BM25 (baseline) | 0.805 | 0.127 | 0 | — | — |
| Random (floor) | 0.037 | 0.461 | 0 | 16% | 9% |

*Precision@10: fraction of top-10 matches sharing the same proximal cause. ECE: calibration error (lower = more trustworthy confidence scores). Correct fix: chose the right remediation from 5 options. Would worsen: chose a remediation that would make things worse.*

**Key findings:** All tested LLMs achieve comparable retrieval precision to BM25 keyword matching (~0.80), but dramatically outperform it on remediation (72-96% correct vs 16% random). GPT-4.1 nano has the best calibration at the lowest token cost. Claude Opus 4 has the highest remediation accuracy (96%). DeepSeek V3 is competitive on retrieval but weaker on remediation (72% correct, 4% harmful).

## Quick start

```sh
docker build -t sifter .

# Generate corpus + run baselines
docker run --rm -v $(pwd)/data:/app/data --entrypoint sifter-generate sifter \
  --output-dir /app/data --prose-pools-dir /app/data/prose_pools --incidents-per-class 50

# Test your model (tasks 2+3, ~$2 for 200 queries)
docker run --rm -v $(pwd)/data:/app/data \
  -e ANTHROPIC_API_KEY=sk-... -e SIFTER_MODEL=claude-haiku-4-5-20251001 \
  --entrypoint sifter-run sifter \
  --corpus /app/data/corpus.json --queries /app/data/query_payloads.json \
  --output-dir /app/data/results --tasks 2,3 \
  --adapter "python /app/examples/anthropic_adapter.py"

# Score
docker run --rm -v $(pwd)/data:/app/data --entrypoint sifter-score sifter \
  --corpus /app/data/corpus.json --queries /app/data/queries.json \
  --results /app/data/results/custom_task2_results.json
```

## Two products

1. **Synthetic benchmark** (CLI) — evaluates models against a labeled synthetic corpus with deterministic scoring. Compare models, prompts, and approaches.
2. **Web labeling tool** — evaluates models against your organization's real incidents with human SRE labels. Confirms a model works on your specific incident patterns.

### Evaluate against your own incidents

```sh
docker compose up -d          # start web + postgres
# visit http://localhost:8080
```

Import incidents as JSON, label pairs side-by-side ("Would knowing about Incident A help respond to Incident B?"), review model performance vs human labels on the dashboard. The tool highlights red herrings — cases where the model is confident but your SREs disagree.

---

# Technical details

## How the benchmark works

### The problem: red herrings

Two incidents can look nearly identical — same error codes, same latency spikes, same affected services — but have completely different proximal causes requiring different fixes. A retry storm and a genuine downstream slowdown both show elevated latency and 5xx errors. A memory leak and a traffic-driven OOM both show rising memory and pod restarts. A code regression and a config regression both show step-function error rates after a change.

A model that matches on symptoms alone will confidently surface these red herrings. An SRE who trusts that suggestion wastes time pursuing the wrong remediation — or worse, applies a fix that makes things worse (scaling up a service that's in a retry storm feeds more capacity to the storm).

### Three evaluation tasks

| Task | What it measures | Tokens/query | Cost for 200 queries |
|------|-----------------|-------------|---------------------|
| **Task 1: Full-corpus retrieval** | Model sees all 1,000 incidents and must rank the most cause-similar ones. Tests both retrieval and reasoning. | ~75,000 | ~$15-50 |
| **Task 2: Reasoning-only** | BM25 pre-filters to top 20 candidates. Model re-ranks. Isolates reasoning from retrieval. | ~2,000 | ~$0.50-2 |
| **Task 3: Remediation** | Given an incident and 5 multiple-choice options, model picks the best fix. Options include the correct fix, a symptom-masker, a harmful fix, and distractors. | ~500 | ~$0.10 |

Task 2 is the recommended starting point — it's 30x cheaper than Task 1 and produces better results (LLMs reason well over a small candidate set but don't add much over keyword matching for raw retrieval).

Select tasks with `--tasks`:

```sh
sifter-run --tasks 2,3 --adapter "..."   # Recommended (~$2)
sifter-run --tasks 1,2,3 --adapter "..."  # Full evaluation (~$50)
sifter-run --tasks 3 --adapter "..."      # Just remediation (~$0.10)
```

The CLI shows estimated token counts before any API calls are made.

### Scoring metrics

All core metrics are deterministic — no LLM-judge in the scoring pipeline.

- **Retrieval@k** — does the top-k contain any correct match? Reported per difficulty tier (easy/medium/hard).
- **Cause-match precision@k** — fraction of top-k sharing the same proximal cause, reported as *lift over class base rate* to normalize for class-frequency imbalance. Raw precision alone is misleading because classes with more incidents have higher precision by chance.
- **Confusion matrix** — which wrong classes dominate predictions? Shows exactly where a model fails (e.g., confusing `downstream_slowdown` with `cascading_timeout`).
- **Calibration ECE** — are the model's confidence scores trustworthy? Bucketed by confidence range. A model that claims 90% confidence should be correct ~90% of the time.
- **Remediation grades** — distribution over {fixed, no-op, masks-symptom, would-worsen}. The *would-worsen* rate is safety-critical and always reported separately — never averaged into a composite.
- **Efficiency** — tokens/query, tokens/correct-match, precision-per-1k-tokens. Enables quality-vs-cost comparison across prompting strategies.

There is no composite headline score. Per-tier vectors are the primary output. This is intentional — composites invite leaderboard chasing and hide the red-herring failure mode that this benchmark exists to measure.

## Corpus design

### Taxonomy: 20 cause classes

| Family | Cause Classes |
|--------|--------------|
| Dependency (6) | retry_storm, downstream_slowdown, dependency_outage, dns_resolution_failure, certificate_expiry, connection_pool_exhaustion |
| Resource (4) | memory_leak, traffic_allocation, cpu_throttling, disk_pressure |
| Deploy (2) | code_regression, config_regression |
| Network (2) | network_partition, load_balancer_misconfiguration |
| Traffic (1) | traffic_spike |
| Amplification (1) | cascading_timeout |
| Data/Schema (2) | schema_migration_failure, data_corruption |
| Timing (2) | clock_skew, race_condition |

Each class has defined **confusable partners** — cause classes that share symptoms but require different remediation. These are assigned difficulty tiers (easy/medium/hard) and scored separately. Examples of hard pairs:

- `retry_storm` vs `downstream_slowdown` — both show latency spikes and 5xx errors, but one is self-inflicted amplification and the other is a genuinely slow dependency
- `memory_leak` vs `traffic_allocation` — both show OOM kills, but one grows monotonically regardless of traffic and the other correlates with request volume
- `code_regression` vs `config_regression` — both show step-function errors after a change, but rolling back the code only fixes one of them

### Taxonomy validation

The taxonomy was validated against three real-world data sources:

- **3,087 cloud provider incidents** from AWS, Azure, and GCP status pages — confirmed coverage of infrastructure-level causes (network, deploy, dependency classes)
- **32 classified incidents from danluu/post-mortems** — confirmed coverage of internal causes (retry storms, race conditions, memory leaks) that cloud providers don't disclose publicly
- **RCAEval fault injection dataset** — confirmed fingerprint patterns for memory_leak and cpu_throttling match real fault behavior

Every cause class has at least one real-world validation source.

### Corpus generation

Each incident consists of a responder summary (2-5 sentences), 5-15 sampled alerts, 20-100 sampled log lines, a topology fragment, and timestamps. Models see this payload; they never see the hidden labels (cause class, fingerprint, remediation vocabulary).

**Why synthetic data?** Real incident corpora are proprietary, unlabeled, and can't provide ground truth for "same proximal cause." Synthetic data lets us control the labels exactly. The risk is that synthetic data has patterns real data doesn't — see "Adversarial style probe" below.

**How incidents are generated:**

1. **Archetype templates** define the structural skeleton for each cause class: fingerprint shape, alert/log patterns, topology
2. **LLM prose pools** provide varied text. 20 summaries, 25 alerts, and 35 log lines per class, generated by Sonnet with real AWS/Azure/GCP incident reports as style examples
3. **Cross-class noise injection** — 20-40% of alerts and logs in each incident come from other cause classes, breaking structural patterns that correlate with cause
4. **Summary style randomization** — 5 structural variants (terse, split, merged, filler, passthrough) prevent sentence-length distributions from leaking cause class

### The circularity problem

A fingerprint schema encodes a theory of what discriminates proximal causes. Generating incidents from the schema and grading against it creates a closed loop — the benchmark could measure how well it matches its own assumptions rather than real-world capability.

**Resolution:** The fingerprint is not part of the model's interface. Models see unstructured payloads (prose, alerts, logs) and must do their own feature extraction. The fingerprint is used only during ground-truth construction. This doesn't eliminate the taxonomy dependence, but it removes the most obvious circular path.

### Adversarial style probe

The biggest risk for any synthetic benchmark: models learn to detect the generator's writing style rather than reasoning about the incident content. If a classifier can predict cause class from *style features alone* (sentence length, punctuation patterns, formatting — not vocabulary), the corpus has a leak.

Sifter runs an adversarial probe on every corpus version:

1. Strip all content from incident text (replace every word with `W`, numbers with `N`, punctuation with `P`)
2. Train a logistic regression classifier on character n-grams and numeric features
3. Gate: summary-only stripped char n-grams must score ≤ random + 20pp

The current corpus passes (19.1% vs 25% threshold). Full-text scores remain elevated (~45%) due to inherent formatting differences in log content across incident types — this is expected and acceptable since log formatting *is* content.

## Plugging in your model

### Subprocess protocol (any language)

Write a program that reads JSON-lines from stdin and writes JSON-lines to stdout:

```python
#!/usr/bin/env python3
import json, sys

for line in sys.stdin:
    req = json.loads(line)
    corpus = req["corpus"]
    k = req["k"]
    matches = [{"incident_id": c["incident_id"], "confidence": 0.5} for c in corpus[:k]]
    sys.stdout.write(json.dumps({"ranked_matches": matches}) + "\n")
    sys.stdout.flush()
```

Report `token_usage` for efficiency metrics:
```json
{"ranked_matches": [...], "token_usage": {"input_tokens": 5000, "output_tokens": 200}}
```

### Pre-built adapters

| Adapter | Providers | Auth env var |
|---------|-----------|-------------|
| [`anthropic_adapter.py`](examples/anthropic_adapter.py) | Anthropic (Claude) | `ANTHROPIC_API_KEY` |
| [`openai_compat_adapter.py`](examples/openai_compat_adapter.py) | OpenAI, Groq, Together, Fireworks, DeepSeek, OpenRouter, Ollama, vLLM, llama.cpp | `OPENAI_API_KEY` + `OPENAI_BASE_URL` |
| [`gemini_adapter.py`](examples/gemini_adapter.py) | Google Gemini (AI Studio) | `GEMINI_API_KEY` |
| [`bedrock_adapter.py`](examples/bedrock_adapter.py) | AWS Bedrock (all models) | AWS credentials + `AWS_REGION` |

All adapters use `SIFTER_MODEL` env var to select the model.

### Python adapter

```python
from sifter_bench.harness import Adapter
from sifter_bench.models import IncidentPayload, RankedMatch

class MyModel(Adapter):
    def retrieve(self, query: IncidentPayload, corpus: list[IncidentPayload], k: int = 10):
        return [RankedMatch(incident_id=c.incident_id, confidence=0.5) for c in corpus[:k]]
```

### Running benchmarks across many models

Use OpenRouter (one API key for all providers) to sweep multiple models:

```sh
export OPENROUTER_API_KEY=sk-or-...
docker run --rm -v $(pwd)/data:/app/data -e OPENROUTER_API_KEY \
  --entrypoint python sifter \
  scripts/benchmark_models.py --data-dir /app/data --tasks 2,3
```

## Web labeling tool details

### How it works

1. **Import** incidents as JSON — minimum fields: `id`, `summary`, `timestamp`. Optional: `severity`, `service`, `url` (link to your incident tracker)
2. **Pairs are sampled** across the model's confidence spectrum (high/medium/low/random) to calibrate both true positives and false positives
3. **SREs label** pairs: "Would knowing about Incident A help you respond to Incident B?" — Yes / No / Maybe
4. **Dashboard** shows precision/recall at multiple thresholds, calibration curves, inter-annotator agreement, and red herring detection

### Deployment

```sh
# Production (PostgreSQL + Gunicorn)
docker compose up -d

# Local dev (SQLite)
docker compose up web-dev
```

For AWS: push the image to ECR, run on Fargate, point `DATABASE_URL` at RDS PostgreSQL, put an ALB in front using the `/health` endpoint. The same `docker-compose.yml` works on any Docker host.

## What this does NOT measure

These limitations are fundamental, not bugs to be fixed:

- **Production readiness.** Synthetic incidents don't capture the messiness of real incident data. Use the web labeling tool for production validation.
- **Domain breadth.** The taxonomy covers cloud-native Kubernetes HTTP microservices only. Mainframes, data pipelines, embedded systems, and non-HTTP architectures are out of scope.
- **Multi-turn investigation.** All tasks are single-turn. Real incident response involves iterative investigation — this benchmark doesn't measure that.
- **Taxonomy disagreement.** If a model's understanding of what constitutes a "proximal cause" differs from the taxonomy's, the model is graded as wrong whether or not it actually is. The taxonomy is the benchmark's ground truth, not objective truth.
- **Distribution shift.** Models trained on the published corpus will overfit. The private query slice helps detect this, but the benchmark has a finite useful life before training-set contamination makes scores meaningless.

## Design rationale

The full design document is in [`sifter-v1-spec.md`](sifter-v1-spec.md), including:

- 7 failure modes pressure-tested during design
- Why "proximal cause" and not "root cause"
- Why there's no composite headline score
- The org-specific tool design (Phase 6)
- Build plan with exit gates for each phase

## License

MIT
