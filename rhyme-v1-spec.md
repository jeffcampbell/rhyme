# Rhyme v1 — Spec & Pressure-Test

*A benchmark for evaluating LLM-based correlation of microservice incidents, with explicit treatment of the red-herring problem.*

---

## 0. Background & motivation

An "AI SRE" tool that surfaces similar past incidents to responders has a high risk of producing red herrings: incidents often share symptoms while having completely different proximal causes. A tool that confidently surfaces a symptom-similar-but-cause-different incident can actively harm response (wrong remediation applied, real cause missed).

This document specifies a reproducible benchmark for measuring how well models can identify *cause-similar* (not merely symptom-similar) incidents in a controlled synthetic corpus.

**Scope split.** This work is one of two products identified during planning:

1. **Rhyme (this document).** A generic, model-agnostic benchmark using synthetic data. Compares models in a vacuum.
2. **Org-specific refinement tool (tabled).** A tool that ingests an organization's real incident corpus, generates synthetic incidents from it, and lets internal SREs vote on correlation quality to refine model performance on their specific domain. Not specified here.

---

## 1. What the benchmark measures

Given a query incident, a model's ability to:

- **(Core)** Retrieve past incidents sharing the same *proximal cause*, not just similar symptoms.
- **(Optional)** Propose a remediation that would actually apply to the query incident.

Scored separately by confusability tier (easy / medium / hard) so that strong performance on easy cases does not mask poor performance on the cases that matter most.

## 2. What it explicitly does not measure

These limitations must ship prominently with the benchmark, not buried in a footnote:

- Production readiness or performance on real incident data.
- Performance on domains other than cloud-native HTTP microservices.
- Agentic or multi-turn investigation.
- Calibration under distribution shift.
- Anything the cause-class taxonomy doesn't represent — models that disagree with the taxonomy are graded as wrong, whether or not they are actually wrong.

---

## 3. Design journey — key decisions and why

These are recorded so that future maintainers understand the rationale, not just the spec.

### 3.1 The fingerprint schema (kept, repurposed)

Early work defined a fingerprint schema across six axes: golden signals, error-class distribution, topology locus, temporal pattern, event correlation, and log keywords. Amplification signal (retry storms vs. real downstream slowdowns) was folded into golden signals and topology as sub-fields, with a derived "amplification score" used as a confusability tiebreaker.

**Discriminability check findings** (walked on four confusable pairs):

- *Retry storm vs. genuine downstream slowdown:* well-discriminated across four axes.
- *DB slow query vs. connection pool exhaustion:* discriminable only if the fingerprint includes DB-side saturation metrics and log keywords. Edge-only telemetry cannot split these.
- *Code regression vs. config regression:* weakly discriminable. Requires deploy-system instrumentation (`deploy.code` vs `deploy.config`) that most orgs don't emit. Accept as a known limitation.
- *Memory leak vs. traffic-driven allocation:* well-discriminated, but only if the temporal-pattern axis has sub-fields (onset shape + traffic correlation + duration), not a single bucket.

### 3.2 The circularity problem

A schema-driven eval has a structural risk: if the generator, the schema, and the evaluator share assumptions, the benchmark measures itself. The fingerprint schema encodes a theory of what discriminates proximal causes; generating incidents from the schema and grading against it closes the loop.

**Resolution adopted for this benchmark.** The fingerprint is not part of the model-under-test's interface. Models see unstructured-ish payloads (prose summary, sampled alerts, sampled log lines, topology fragment) and must do their own feature extraction. The fingerprint is used only during ground-truth construction — it tells us, the benchmark authors, which incidents share proximal cause. This does not eliminate the taxonomy dependence (see §10, failure mode 2), but it removes the most obvious circular path.

### 3.3 Why "benchmark" and not "platform"

A benchmark has different constraints than a tool: reproducibility, portability, deterministic scoring, model-agnostic task shape. The org-specific refinement tool was tabled precisely because it has the opposite constraints (tailored, interactive, org-data-specific). Trying to be both would compromise both.

---

## 4. Corpus

- **Size:** target ~1,000 labeled synthetic incidents (floor 500, ceiling 2,000). Final size determined empirically by when cause-match precision@k stabilizes across query sampling.
- **Generation:** hybrid pipeline — archetype templates define structure and labels; LLM fills prose, log realism, surface variation. Every generated incident is validated against its fingerprint before admission to the corpus.
- **Taxonomy:** ~20 cause classes across ~8 families (deploy, resource, dependency, network, traffic, retry/amplification, data/schema, timing). Versioned and shipped with the benchmark. See §10 failure mode 2 for validation requirements.
- **Domain:** cloud-native Kubernetes-based HTTP microservices. Explicit and bounded.
- **Per-incident record:**
  - *Model-visible payload:* responder summary (2–5 sentences), 5–15 sampled alerts, 20–100 sampled log lines, small topology fragment, timestamps.
  - *Hidden labels:* proximal-cause class, remediation that worked, remediation-that-masks (the "looks like it helped but didn't" case), confusability tier, fingerprint vector.

## 5. Query set

- 200 held-out query incidents, stratified ~30% easy / ~40% medium / ~30% hard by confusability tier.
- Each query has a ground-truth set of "correct matches" (same proximal-cause class) and "tempting wrong matches" (high symptom similarity, different proximal cause).
- Additional held-out **private slice** of ~50 queries. Never published. Run only by the benchmark maintainer for leaderboard verification. Guards against training-set contamination.

---

## 6. Tasks

All tasks are single-turn in v1. Agentic / multi-turn deferred.

### Task 1 — Retrieval + ranking (core)
- **Input:** query incident payload + full corpus (open-corpus mode).
- **Output:** ranked list of top-10 matches with confidence scores 0–1.

### Task 2 — Reasoning-only retrieval
- **Input:** query + pre-retrieved top-20 candidates from a neutral BM25 retriever (closed-corpus mode).
- **Output:** re-ranked top-10 with confidence scores.
- **Purpose:** isolates reasoning from retrieval quality. Distinguishes "the model's retriever is bad" from "the model's judgment is bad."

### Task 3 — Remediation suggestion (optional, see §10 failure mode 4)
- **Input:** query + model's top-3 retrieved matches.
- **Output:** proposed remediation in natural language.
- **Grading:** deterministic mapping against controlled remediation vocabulary linked to proximal-cause class.
- **Open issue:** controlled-vocab grading is brittle. Decision required before build: drop, accept LLM-judge with non-determinism caveat, or reformulate as multiple-choice.

---

## 7. Metrics

All core metrics are deterministic. No LLM-judge in the headline.

- **Retrieval@k (per tier):** does top-k contain any same-cause-class incident?
- **Cause-match precision@k (per tier):** fraction of top-k sharing proximal cause. **Primary red-herring metric.** Reported as *lift over class base rate* to normalize for class-frequency imbalance (see §10 failure mode 3).
- **Confusion-matrix delta:** for each query class, which wrong classes dominate top-k? Reported as a matrix, not averaged. Diagnostic rather than scalar — tells you *where* a model fails.
- **Calibration ECE:** bucket predictions by confidence, check cause-match precision per bucket. Critical because the tool outputs confidence scores and responders will lean on them.
- **Remediation grade (Task 3):** distribution over {fixed, no-op, masks-symptom, would-worsen}. The "would-worsen" rate is safety-critical and must never be averaged into a composite.

### On composite scores

The spec reserves a weighted composite "headline" score for leaderboard sorting, but its inclusion is an open question. Composites invite leaderboard chasing and hide tradeoffs. Strongly consider shipping *no* composite and requiring consumers to report the full per-tier vector. Revisit before release.

---

## 8. Baselines (shipped with the benchmark)

A benchmark without baselines produces meaningless numbers. All of these ship in v1:

1. Random retrieval (floor).
2. BM25 over full payload.
3. Embedding cosine with a standard off-the-shelf open embedding model (version frozen).
4. LLM-as-retriever with a small open model, zero-shot prompt.
5. LLM-as-retriever with a small open model, schema-prompt variant (told what fingerprint dimensions to attend to).

Baseline 4 is the critical reference point — it shows whether LLM reasoning beats pure lexical/vector retrieval at all.

---

## 9. Reproducibility contract

- Corpus, taxonomy, query set, baselines, and scoring code are versioned and published together as `rhyme-v1.0`.
- Scoring is deterministic given model outputs. Two people scoring the same output file must get the same number.
- Model outputs for any submission are retained (not just scores) so they can be re-verified or re-scored under a future metric version.
- Benchmark version bumps (v1.1, v2.0) are additive. Historical v1.0 scores remain valid forever.
- Private held-out slice is rotated if training-set contamination is suspected.

### User workflow

1. Download `rhyme-v1.0.tar`.
2. Implement the adapter interface: `retrieve(query_payload, corpus) -> ranked_list_with_scores`; optionally `remediate(query, top_matches) -> string`.
3. Run the provided harness, which issues queries and collects outputs.
4. Run the provided scorer, which produces a report: headline (if kept), per-tier breakdown, confusion matrix, calibration chart, baseline comparison.
5. Optionally submit outputs for private-slice verification.

---

## 10. Pressure-test: failure modes

Ranked by the author's assessment of risk. The first three need real thinking and could kill v1.

### Failure mode 1 — Corpus realism collapse (high severity, high likelihood)

**The failure.** LLM-generated incidents feel real but have statistical regularities (phrasing, sentence length, vocabulary) that correlate with cause class because the generator has priors. Models-under-test learn to detect the generator's hand rather than underlying cause. Scores climb; benchmark becomes meaningless. Most common failure mode for synthetic benchmarks.

**Mitigation.** Multiple generators (rotate 2–3 LLMs for prose). Style-randomization post-processing. Explicit adversarial probe: train a small classifier to predict cause class from *style features only* (n-grams, sentence length, punctuation). If that classifier beats random, the corpus has a leak. Run on every corpus version.

### Failure mode 2 — Taxonomy is load-bearing and wrong (high severity, high likelihood)

**The failure.** The ~20 cause-class taxonomy is ground truth for "same proximal cause." If the taxonomy cuts at wrong joints, every score is wrong in consistent-looking ways. Insidious because the ground truth *is* the taxonomy — probes cannot detect it.

**Mitigation.**
- Validate the taxonomy against real incident postmortems from public sources (AWS, Cloudflare, GitHub status) before freezing. Do real incidents fit into exactly one class? If many don't, the taxonomy is wrong.
- Ship a "taxonomy confidence" annotation per incident. Prototypical examples vs. borderline. Score borderline cases with partial credit or exclude from the headline.
- The tabled org-specific tool becomes the strongest long-term validator — if adopting orgs keep overriding the taxonomy, v1 got it wrong.

### Failure mode 3 — Red-herring metric dominated by class frequency (medium-high severity)

**The failure.** Cause-match precision@k depends on how many same-cause incidents exist in the corpus. Classes with 3 incidents have max precision 0.3 at k=10 regardless of model quality. Classes with 200 make random retrieval look good.

**Mitigation.** Normalize precision@k against class frequency — report "precision lift over class base rate" rather than raw precision. Alternatively, set k per-query to class size, capped at 10. Either way, the naive metric will mislead.

### Failure mode 4 — Task 3 ungradable (medium severity, certain)

**The failure.** NL-to-controlled-vocabulary mapping is always brittle. "Restart the service," "bounce the pod," "rolling restart" should map to one concept. Vocabulary tight → false negatives. Vocabulary loose → false positives.

**Mitigation options (decision required):**
- (a) Drop Task 3 from v1. Honest, costs the most operationally relevant signal.
- (b) Accept LLM-judge for Task 3 with explicit non-determinism caveat. Realistic, breaks the reproducibility contract for this sub-metric.
- (c) Reformulate as multiple-choice — model picks from 5 candidate remediations. Deterministic, less realistic.

No clean answer. Make a real decision, don't punt.

### Failure mode 5 — Goodhart (medium severity, near-certain over time)

**The failure.** Frontier labs train on published benchmarks. Scores saturate without real capability gains within ~12 months of release. The private slice helps only if it's genuinely distributionally distinct, not just "same generator, different seed."

**Mitigation.** Plan version churn from day one. New corpus generation (different archetypes, different confusability pairs) every 12–18 months. Private slice uses different generator prompts and possibly a different archetype subset. Treat v1 as having a finite useful life.

### Failure mode 6 — Scope is wrong (medium severity)

**Version A:** Microservices too narrow — evaluators with mainframes, data pipelines, embedded systems find no value. Adoption suffers.

**Version B:** "Microservices" too broad — Kubernetes-native and legacy-ESB shops have wildly different incident patterns. Scores don't generalize.

**Mitigation.** For A: accept for v1, plan domain expansion. For B: the v1 scope is already constrained to "cloud-native Kubernetes-based HTTP microservices" per §4. Document explicitly.

### Failure mode 7 — Nobody runs it (existential for impact, not validity)

**The failure.** Benchmarks that aren't championed by a major lab, hosted with a public leaderboard, or connected to an existing community tend to die regardless of quality.

**Mitigation (not primarily technical).** Partnerships (Anthropic, observability vendors, SRE conferences). Public leaderboard hosting. Visible first-runs from known labs. Consider an IBM-internal angle: build it as a tool IBM uses to select models for internal SRE tooling, publish as a side-effect, don't depend on external adoption for success.

---

## 11. Build effort (rough)

Order-of-magnitude, not commitment:

| Work item | Estimate |
|---|---|
| Taxonomy + archetype library (hard design work) | 1–2 weeks |
| Corpus generation pipeline with validation loop | 2–3 weeks |
| Query set construction + confusability stratification | 1 week |
| Scoring code + baselines + harness | 2 weeks |
| Documentation, private slice, release packaging | 1 week |
| **Realistic total** | **~2 months focused, likely longer** |

Corpus quality is the long pole. Generating 1,000 incidents that are realistic *and* correctly labeled tends to blow up in practice.

---

## 12. Recommended next step: a 50-incident prototype

Before committing to the ~2-month v1 build, produce a prototype corpus: ~50 hand-built incidents across ~5 cause classes, graded by 2–3 SREs on taxonomy fit, run against the baselines.

**Cost:** ~2 weeks.

**What it tests cheaply:**

- Failure mode 2 (taxonomy fit) — if the taxonomy fails on 50 hand-built incidents, it will fail on 1,000 generated ones.
- Failure mode 3 (metric design) — if baselines can't be meaningfully differentiated at n=50, the metric needs rework before scaling.
- Failure mode 4 (Task 3 gradability) — try the controlled-vocab grader on 50 remediations, see if it breaks.

**Decision point after prototype:** commit to v1 build, iterate on taxonomy/metrics, or reconsider approach.

---

## 13. Open questions — resolved

Decisions made during Phase 2 of the v1 build. Rationale documented inline.

1. **Composite headline score — DROPPED.** The scorer reports per-tier vectors as the primary output. A macro-average is included as a reference but labeled "not a headline score." Rationale: composites invite leaderboard chasing and hide the red-herring problem that this benchmark exists to measure. A model that scores well on easy cases but fails on hard confusable pairs should not be masked by a high composite.

2. **Task 3 grading — (c) MULTIPLE-CHOICE.** Each query gets 5 options: the canonical fix, the masks-symptom fix, the would-worsen action (where applicable), and 2 plausible distractors (canonical fixes from other cause classes). Grading is fully deterministic: compare selected label to the known grade. The `would_worsen` rate is always reported separately and never composited. Rationale: the prototype's keyword-based free-text grader was too brittle (spec §10 FM4). LLM-judge breaks the reproducibility contract. Multiple-choice preserves determinism while remaining operationally meaningful.

3. **DB-side metrics — SYNTHESIZED.** Three optional fields added to the fingerprint: `db_query_time_ms`, `db_connection_pool_utilization`, `db_lock_contention`. Populated for classes where DB interaction is discriminating: `downstream_slowdown` (slow queries, normal pool), `connection_pool_exhaustion` (fast queries, maxed pool), `schema_migration_failure`, `data_corruption`, `race_condition` (elevated lock contention). Rationale: the cost of synthesis is modest and it enables the `connection_pool_exhaustion` cause class that was otherwise indiscriminable from `downstream_slowdown`.

4. **Deploy instrumentation — SYNTHESIZED.** A `deploy_type` field added to the fingerprint: `"code"` for `code_regression`, `"config"` for `config_regression`, `null` for all other classes. Rationale: real-world discriminability is lower than synthetic (most orgs don't emit `deploy.code` vs `deploy.config` distinctly), but synthesizing it enables the pair. Incidents in this pair are annotated with lower taxonomy confidence as a known limitation.

5. **Corpus size — DEFERRED to Phase 4.** Will be validated empirically by plotting precision@k variance across query samplings at 500, 750, 1000, 1500 incidents. Stop when variance stabilizes (<0.02 change between sizes).

6. **Adoption strategy — DEFERRED.** Does not block the technical build. To be decided before Phase 5 (release packaging).

---

## 14. Concrete v1 build plan

*Added after the 50-incident prototype was completed and validated. This section replaces the rough estimates in §11 with a phased plan informed by prototype findings.*

### 14.0 Prototype findings (context for the plan)

The prototype (50 incidents, 5 cause classes, 3 baselines) confirmed:

- **Taxonomy fit (FM2):** 5 classes across 3 families produce distinguishable incidents. Confusable pairs (retry_storm ↔ downstream_slowdown, memory_leak ↔ traffic_allocation) appear correctly in confusion matrices. The taxonomy works at this scale.
- **Metric design (FM3):** Baselines are clearly differentiated — Random: 0.14 precision@10, TF-IDF: 0.50, BM25: 0.80. Lift normalization works correctly.
- **Corpus realism (FM1) — the problem:** BM25 at 0.80 precision@10 is far too high for a keyword baseline. Template-generated incidents share vocabulary within cause classes. This is the highest-priority problem for v1 and shapes the plan below.
- **Task 3 grading (FM4):** The prototype's keyword overlap grader is too brittle. A real decision is needed (see Phase 2).

### 14.1 Phase 1 — Expand taxonomy to v1 target (~20 cause classes)

**Goal:** Finalize the taxonomy before generating the full corpus, because taxonomy changes invalidate everything downstream.

**Tasks:**

1. **Expand from 5 to ~20 cause classes across ~8 families.** Proposed additions (validate before committing):
   - *Deploy family:* config_regression (distinct from code_regression — requires deploy-system instrumentation, see §13 Q4)
   - *Resource family:* cpu_throttling, disk_pressure
   - *Dependency family:* dns_resolution_failure, certificate_expiry, connection_pool_exhaustion
   - *Network family:* network_partition, load_balancer_misconfiguration
   - *Traffic family:* traffic_spike (organic), bot_traffic
   - *Retry/amplification family:* cascading_timeout (distinct from retry_storm — no retry amplification, just serial timeout propagation)
   - *Data/schema family:* schema_migration_failure, data_corruption
   - *Timing family:* clock_skew, race_condition

2. **Validate against public postmortems.** Take 30+ real postmortems from AWS, Cloudflare, GitHub, and Google status pages. Each must fit exactly one cause class. Track fit rate — if <80% fit cleanly, the taxonomy needs rework. Document borderline cases.

3. **Define all confusable pairs with tier assignments.** Every new class needs at least one confusable partner. Map the full matrix. This directly determines the hard-tier query set.

4. **Write remediation vocabulary for every class.** Each class needs: canonical fix, masks-symptom fix, and (where applicable) would-worsen action. These must be distinct enough for grading.

5. **Write archetype templates for every class.** Alert templates, log templates, summary templates, fingerprint shape. Follow the pattern in `archetypes.py`.

**Exit gate:** The taxonomy is frozen after this phase. Every cause class has been validated against real incidents, has archetype templates, has remediation vocabulary, and has confusable-pair assignments. Document the frozen taxonomy version as `v1.0.0`.

### 14.2 Phase 2 — Resolve open decisions from §13

**Goal:** Make the deferred decisions so they don't block implementation.

Resolve each of these and document the decision + rationale in this spec (edit §13 inline):

1. **Composite headline score.** Recommendation: drop. Ship per-tier vectors only. A composite invites gaming and hides the red-herring failure mode that this benchmark exists to measure.

2. **Task 3 grading.** Recommendation: (c) multiple-choice. Reformulate as: given the query incident and top-3 matches, select the best remediation from 5 candidates (canonical, masks-symptom, would-worsen, and 2 plausible-but-wrong distractors). This is deterministic, gradable, and avoids the NL mapping problem. The controlled-vocab keyword grader from the prototype is insufficient.

3. **DB-side metrics.** Recommendation: synthesize them. Add DB-specific fields to the fingerprint (query_time, connection_pool_utilization, lock_contention). The cost is modest and it enables the connection_pool_exhaustion cause class.

4. **Deploy instrumentation.** Recommendation: synthesize `deploy.code` vs `deploy.config` as an event correlation field in the fingerprint. Accept that real-world discriminability is lower than synthetic discriminability and annotate these incidents with lower taxonomy confidence.

5. **Corpus size.** Validated empirically in Phase 4.

6. **Adoption strategy.** Defer — does not block the technical build.

**Exit gate:** All technical decisions (1–4) are resolved and documented.

### 14.3 Phase 3 — Corpus generation pipeline with realism controls

**Goal:** Build the pipeline that produces ~1,000 correctly-labeled incidents without the lexical regularity problem.

**Tasks:**

1. **Multi-LLM prose generation.** The archetype templates provide structure (fingerprint, alert shapes, topology). Feed these to 2–3 different LLMs (e.g., Claude, GPT-4, Llama) to generate the prose fields (summary, alert messages, log messages). Each incident is generated by one LLM, assigned round-robin. The structural labels come from the template; the surface text comes from the LLM.

2. **Style randomization post-processing.** After LLM generation, apply randomization:
   - Vary sentence count in summaries (2–5, per spec)
   - Randomize log line count (20–100)
   - Shuffle alert ordering (not always chronological — real alerts arrive out of order)
   - Inject noise: irrelevant alerts from unrelated services, debug-level log lines from healthy services
   - Vary naming conventions (camelCase vs snake_case service names, different timestamp formats)

3. **Fingerprint-to-incident validation.** After generation, validate each incident against its fingerprint. If the fingerprint says `amplification_ratio > 3.0`, the generated alerts and logs must actually show amplified request rates. Reject and regenerate incidents that don't match. This is the quality gate.

4. **Adversarial style probe (mandatory gate).** Train a lightweight classifier (logistic regression on character n-grams + sentence length features) to predict cause class from incident text. Run on every corpus version. If accuracy exceeds `1/num_classes + 0.10` (i.e., >15% for 20 classes), the corpus has a stylistic leak. Diagnose which features leak and fix the generation pipeline. This gate blocks release.

5. **Per-incident metadata.** Tag each incident with: generator LLM, generation seed, taxonomy confidence (prototypical vs borderline), generation timestamp.

**Exit gate:** 1,000+ incidents pass fingerprint validation AND the adversarial style probe. BM25 baseline precision@10 drops to <0.40 (below the prototype's 0.80).

### 14.4 Phase 4 — Query set, scoring, and baselines at scale

**Goal:** Build the full evaluation infrastructure around the v1 corpus.

**Tasks:**

1. **Query set construction.** 200 held-out queries, stratified ~30% easy / ~40% medium / ~30% hard per spec §5. Plus ~50 private-slice queries generated with different generator prompts (not just different seeds — structurally different archetype variants).

2. **Corpus size validation.** Generate corpora at 500, 750, 1000, 1500 incidents. Run BM25 and TF-IDF baselines on each. Plot precision@k variance across 5 random query samplings. Stop at the size where variance stabilizes (change <0.02 between sizes).

3. **Task 2 implementation (reasoning-only retrieval).** Implement the BM25 pre-retriever that returns top-20 candidates. Build the closed-corpus harness variant. This isolates model reasoning from retrieval quality.

4. **Task 3 implementation (multiple-choice remediation).** Generate the 5-option multiple-choice sets per query. Implement deterministic grading. Track `would_worsen` rate separately — it must never be composited.

5. **LLM baselines (4 and 5 from §8).** Implement LLM-as-retriever adapters:
   - Baseline 4: small open model (e.g., Llama 3), zero-shot prompt, full payload
   - Baseline 5: same model, schema-aware prompt (told what fingerprint dimensions to attend to, but not given the actual fingerprint values)
   - These require API access and are non-deterministic — run 3x and report mean ± std.

6. **Calibration analysis.** Verify ECE computation at scale. Produce calibration plots (bucketed confidence vs actual precision). If models are systematically overconfident, note this in the benchmark documentation.

**Exit gate:** All 5 baselines produce clearly differentiated scores. Per-tier breakdown shows hard-tier scores meaningfully lower than easy-tier. Private-slice scores are within ±0.05 of public-slice scores (if not, the slices are distributionally different, which is a problem).

### 14.5 Phase 5 — Packaging and release (Product 1: Synthetic Benchmark)

**Goal:** Ship `rhyme-v1.0` as a reproducible, self-contained benchmark with a generic model connector.

**Tasks:**

1. **Subprocess connector protocol.** A language-agnostic way to plug in any model:
   - The harness spawns a subprocess, writes query JSON to stdin, reads ranked results from stdout.
   - One query per line (JSON-lines format). The subprocess reads a line, writes a line.
   - Schema: input is `{"query": <IncidentPayload>, "corpus": [<IncidentPayload>, ...], "k": 10}`. Output is `{"ranked_matches": [{"incident_id": "...", "confidence": 0.9}, ...], "token_usage": {"input_tokens": N, "output_tokens": N}}`.
   - Token usage is optional. Confidence scores are required.
   - A reference implementation wrapping `curl` to an OpenAI-compatible API is included as an example.

2. **Adapter interface documentation.** Clear docs for both the Python `Adapter` class and the subprocess protocol. Include worked examples.

3. **Reproducibility verification.** Two people independently run the same model outputs through the scorer and get identical numbers. Fix any floating-point or ordering nondeterminism.

4. **Release tarball contents:**
   - Corpus JSON (without private slice)
   - Query set JSON (public slice)
   - Query payloads JSON (model-visible portion)
   - Remediation questions JSON
   - Scoring code
   - Harness code with subprocess connector
   - Baseline implementations and reference scores
   - Taxonomy definition (versioned)
   - Dockerfile

5. **Documentation:**
   - What the benchmark measures and explicitly does not measure (from §1, §2)
   - Known limitations and failure modes (from §10)
   - Taxonomy with validation evidence
   - Baseline reference scores with interpretation guide
   - Efficiency metrics interpretation

6. **Version freeze.** Tag as `v1.0.0`. All artifacts are immutable from this point. Future work is v1.1 or v2.0.

**Exit gate:** The tarball can be downloaded, built, and run by someone with no prior context, producing correct baseline scores via both the Python adapter and subprocess connector.

### 14.6 Phase 6 — Org-specific correlation tool (Product 2)

**Goal:** A web-based tool that lets organizations evaluate model correlation quality against their own real incident data, validated by human SRE judgment.

**Tasks:**

1. **Incident import.** JSON import of real incidents with a minimal schema:
   ```json
   {"incidents": [
     {"id": "INC-123", "summary": "...", "timestamp": "...", "severity": "...", "service": "...", "metadata": {}},
     ...
   ]}
   ```
   No synthetic logs/alerts required — the summary and metadata are sufficient for correlation.
   Future: incident.io MCP server integration for automatic import.

2. **Pair sampling.** Smart selection of incident pairs for human labeling:
   - Run the model connector (from Product 1) against the imported corpus to get correlation scores.
   - Sample pairs across the confidence spectrum: high-confidence matches (verify true positives), low-confidence non-matches (verify true negatives), and medium-confidence pairs (the interesting boundary).
   - Include random pairs as calibration anchors.
   - Target ~200 labeled pairs for statistical significance.

3. **Labeling interface.** Web app (Flask/FastAPI + HTML):
   - Show two incidents side-by-side.
   - SRE answers: "Would knowing about Incident A help you respond to Incident B?" — Yes / No / Maybe.
   - Optional free-text notes.
   - Track labeler identity for inter-annotator agreement.
   - Progress bar and session management.

4. **Scoring against human labels.** Reuse the scoring infrastructure from Product 1:
   - Compute precision, recall, and calibration against human-labeled pairs.
   - Report agreement rate between model and human labels.
   - Report inter-annotator agreement (if multiple SREs label).
   - Flag cases where the model is confident but humans disagree (potential red herrings).

5. **Dashboard.** Web UI showing:
   - Model performance vs human labels (precision, recall, F1 at various thresholds).
   - Calibration curve (model confidence vs human agreement rate).
   - Confusion clusters: which incident types does the model confuse?
   - Efficiency metrics if token usage is available.
   - Export results as JSON for further analysis.

**Exit gate:** An organization can import 100+ incidents, label ~200 pairs in <2 hours, and get a meaningful report on how well a model correlates their specific incident patterns.

### 14.7 Sequencing and dependencies

```
Phase 1 (taxonomy) ──→ Phase 2 (decisions) ──→ Phase 3 (corpus) ──→ Phase 4 (eval) ──→ Phase 5 (Product 1)
                                                     │                                        │
                                                     └── adversarial probe blocks Phase 4      └──→ Phase 6 (Product 2)
                                                                                                    (shares Adapter + scorer)
```

Phases 1-4 complete. Phase 5 packages Product 1. Phase 6 builds Product 2, reusing the adapter interface, scoring code, and efficiency metrics from Product 1.

---

*End of spec. This document reflects the state of design discussion as of the conversation producing it, updated with the concrete build plan after prototype validation. All decisions are revisable; the pressure-test section (§10) is the load-bearing part and should be revisited whenever the spec changes.*
