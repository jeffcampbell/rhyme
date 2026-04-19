# Rhyme

A benchmark for evaluating LLM-based correlation of microservice incidents, with explicit treatment of the red-herring problem.

Two products:
1. **Synthetic benchmark** — evaluates models against a labeled synthetic corpus with deterministic scoring. CLI-based, runs in Docker.
2. **Org-specific correlation tool** — web UI for evaluating models against real incident data with human SRE labels. PostgreSQL-backed, production-ready via docker-compose.

## Architecture

```
src/rhyme_bench/          # Product 1: Synthetic benchmark
  taxonomy.py                    # 20 cause classes, 8 families, remediation vocabulary, confusable pairs
  models.py                      # Pydantic models: Incident, Corpus, QuerySet, Fingerprint, TokenUsage, etc.
  archetypes.py                  # Per-class archetype templates (alerts, logs, summaries, fingerprint shapes)
  generator.py                   # Corpus generation: archetype templates + LLM prose pools + noise injection
  prose_pools.py                 # LLM-generated prose pool schema and loader
  queries.py                     # Query set construction with tier stratification + private slice + remediation MCQ
  harness.py                     # Adapter interface, Task 1/2/3 runners
  subprocess_adapter.py          # Language-agnostic subprocess connector (JSON-lines on stdin/stdout)
  scorer.py                      # All metrics: retrieval@k, precision@k with lift, confusion matrix, ECE, efficiency
  baselines.py                   # Shipped baselines: Random, BM25, TF-IDF
  style_probe.py                 # Adversarial style probe (spec §10 FM1)
  cli.py                         # CLI: rhyme-generate, rhyme-run, rhyme-score, rhyme-probe

src/rhyme_web/            # Product 2: Web labeling tool
  database.py                    # SQLAlchemy models (PostgreSQL + SQLite fallback)
  models.py                      # Pydantic models for import, pairs, labels, scoring
  pair_sampler.py                # Confidence-stratified pair sampling
  scorer_human.py                # Score model predictions vs human labels
  app.py                         # Flask routes: import, label, dashboard, API, health
  cli.py                         # CLI: rhyme-web
  templates/                     # HTML templates (Jinja2)

data/
  prose_pools/                   # LLM-generated prose pools (20 JSON files, one per cause class)
  real_incidents/                # Mined real incidents (AWS, Azure, GCP, danluu, RCAEval)

examples/
  parse_utils.py                 # Shared JSON parsing for all adapters
  echo_adapter.py                # Minimal subprocess adapter reference
  anthropic_adapter.py           # Anthropic Claude adapter
  openai_compat_adapter.py       # OpenAI-compatible (also Mistral, Groq, Ollama, vLLM, etc.)
  gemini_adapter.py              # Google Gemini (AI Studio)
  bedrock_adapter.py             # AWS Bedrock (Converse API)
  sample_incidents.json          # Sample incidents for testing the web tool

Dockerfile                       # Product 1: benchmark CLI
Dockerfile.web                   # Product 2: web tool (Gunicorn)
docker-compose.yml               # Production deployment (web + PostgreSQL)
```

## How to run

Everything runs in Docker. Never install dependencies on the host.

```sh
# Build
docker build -t rhyme .

# Generate corpus (1000 incidents, 50 per class, with LLM prose pools)
docker run --rm -v $(pwd)/data:/app/data --entrypoint rhyme-generate rhyme \
  --output-dir /app/data --prose-pools-dir /app/data/prose_pools --incidents-per-class 50

# Run shipped baselines (Tasks 1, 2, 3)
docker run --rm -v $(pwd)/data:/app/data --entrypoint rhyme-run rhyme \
  --corpus /app/data/corpus.json --queries /app/data/query_payloads.json \
  --output-dir /app/data/results

# Run a custom model (tasks 2+3 only, saves cost)
docker run --rm -v $(pwd)/data:/app/data \
  -e ANTHROPIC_API_KEY=sk-... -e RHYME_MODEL=claude-haiku-4-5-20251001 \
  --entrypoint rhyme-run rhyme \
  --corpus /app/data/corpus.json --queries /app/data/query_payloads.json \
  --output-dir /app/data/results --tasks 2,3 \
  --adapter "python /app/examples/anthropic_adapter.py"

# Score results
docker run --rm -v $(pwd)/data:/app/data --entrypoint rhyme-score rhyme \
  --corpus /app/data/corpus.json --queries /app/data/queries.json \
  --results /app/data/results/bm25_results.json

# Score with remediation
docker run --rm -v $(pwd)/data:/app/data --entrypoint rhyme-score rhyme \
  --corpus /app/data/corpus.json --queries /app/data/queries.json \
  --results /app/data/results/random_results.json \
  --remediation-results /app/data/results/random_remediation_results.json \
  --remediation-questions /app/data/remediation_questions.json

# Run adversarial style probe
docker run --rm -v $(pwd)/data:/app/data --entrypoint rhyme-probe rhyme \
  --corpus /app/data/corpus.json

# --- Product 2: Web labeling tool ---

# Production (PostgreSQL + Gunicorn)
docker compose up -d                    # start web + postgres
docker compose logs -f web              # watch logs
docker compose down                     # stop

# Local dev (SQLite, no PostgreSQL needed)
docker compose up web-dev

# Environment variables for production:
#   DATABASE_URL=postgresql://user:pass@host:5432/dbname
#   SECRET_KEY=<random-hex-string>
```

## Connecting your model

Two options:

### Option A: Subprocess adapter (any language)
Write a program that reads JSON-lines from stdin and writes JSON-lines to stdout. Each line is one task:

**Input (retrieve):**
```json
{"task": "retrieve", "query": {...}, "corpus": [{...}, ...], "k": 10}
```

**Output:**
```json
{"ranked_matches": [{"incident_id": "INC-...", "confidence": 0.9}], "token_usage": {"input_tokens": 1000, "output_tokens": 200}}
```

Token usage is optional. Pre-built adapters in `examples/`: `anthropic_adapter.py`, `openai_compat_adapter.py`, `gemini_adapter.py`, `bedrock_adapter.py`.

Run with: `rhyme-run --tasks 2,3 --adapter "python my_model.py"`

### Option B: Python adapter
Subclass `Adapter` from `harness.py` and implement `retrieve()`. Optionally implement `remediate()`. Return `RetrieveOutput` to include token usage.

## Guardrails

### Scoring must stay deterministic
No LLM-judge in headline metrics. All core metrics (retrieval@k, precision@k, confusion matrix, calibration ECE) must produce identical results given identical inputs.

### Models never see labels or fingerprints
Models receive `IncidentPayload` only — never `IncidentLabels` or `Fingerprint`. This is the circularity problem from spec §3.2.

### Corpus realism — adversarial style probe is mandatory
Run `rhyme-probe` on every corpus version. The probe trains a classifier on style features (content-stripped char n-grams, sentence length) to detect cause-class leakage. The primary gate is on summary prose (stripped char n-grams ≤ random + 20pp). Full-text scores remain elevated due to inherent log format differences — this is expected.

### Taxonomy changes invalidate all scores
Changing a cause class changes ground truth. Validate against real postmortems before modifying. Major version bump required.

### Precision@k must be reported as lift
Normalize against class base rate. The scorer does this — don't bypass it.

### "Would-worsen" rate is safety-critical
Never average into a composite score. Always report separately.

### Efficiency metrics are opt-in
Adapters can return `TokenUsage` to enable quality-vs-cost comparison. The scorer reports tokens/query, tokens/correct-match, and precision-per-1k-tokens.

## Resolved decisions (spec §13)

1. **Composite score** — DROPPED. Per-tier vectors only.
2. **Task 3 grading** — Multiple-choice (5 options, deterministic).
3. **DB metrics** — Synthesized (query_time, pool_utilization, lock_contention).
4. **Deploy type** — Synthesized (code vs config in fingerprint).
5. **Corpus size** — 1000 incidents (50/class). Variance stabilized at ±0.009.
6. **Adoption strategy** — Deferred.

## Code conventions

- Python 3.12+, Pydantic v2 for data models, SQLAlchemy for web tool storage
- All data serialized as JSON
- Type hints on all public functions
- Tests in `tests/`, run with pytest (112 tests)
- Lint with ruff (line-length 100)
- The `Adapter` ABC in `harness.py` and `SubprocessAdapter` in `subprocess_adapter.py` are the public interfaces — keep them minimal and stable
- Web tool uses `DATABASE_URL` env var: PostgreSQL for production, SQLite for local dev
- Never install dependencies on the host — everything runs in Docker
