"""CLI entry points for the benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def generate():
    """Generate corpus, query set, and remediation questions."""
    parser = argparse.ArgumentParser(description="Generate Rhyme corpus")
    parser.add_argument(
        "--incidents-per-class", type=int, default=10,
        help="Number of incidents per cause class (default: 10)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output-dir", type=str, default="data",
        help="Output directory (default: data)",
    )
    parser.add_argument(
        "--prose-pools-dir", type=str, default=None,
        help="Directory with LLM-generated prose pool JSON files (default: <output-dir>/prose_pools)",
    )
    args = parser.parse_args()

    from .generator import generate_corpus
    from .queries import build_query_set, build_private_slice, build_remediation_questions

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    pools_dir = Path(args.prose_pools_dir) if args.prose_pools_dir else output / "prose_pools"

    print(f"Generating corpus with {args.incidents_per_class} incidents per class...")
    corpus = generate_corpus(
        incidents_per_class=args.incidents_per_class,
        seed=args.seed,
        prose_pools_dir=pools_dir,
    )
    corpus_path = output / "corpus.json"
    corpus.save(corpus_path)
    print(f"  Wrote {len(corpus.incidents)} incidents to {corpus_path}")

    # Determine query count based on corpus size
    total_queries = min(200, len(corpus.incidents) // 5)
    print(f"Building query set ({total_queries} queries, stratified 30/40/30)...")
    query_set, query_incidents = build_query_set(corpus, total_queries=total_queries)
    query_path = output / "queries.json"
    query_set.save(query_path)

    tier_counts = {}
    for q in query_set.queries:
        tier_counts[q.confusability_tier.value] = tier_counts.get(q.confusability_tier.value, 0) + 1
    print(f"  Wrote {len(query_set.queries)} queries to {query_path}")
    print(f"  Tier distribution: {tier_counts}")

    # Save query payloads separately (what models see)
    query_payloads_path = output / "query_payloads.json"
    payloads = [qi.payload.model_dump() for qi in query_incidents]
    query_payloads_path.write_text(json.dumps(payloads, indent=2))
    print(f"  Wrote query payloads to {query_payloads_path}")

    # Build private slice
    public_ids = {q.query_id for q in query_set.queries}
    private_count = min(50, (len(corpus.incidents) - len(public_ids)) // 5)
    if private_count > 0:
        private_set, private_incidents = build_private_slice(corpus, public_ids, total_queries=private_count)
        private_path = output / "queries_private.json"
        private_set.save(private_path)
        private_payloads_path = output / "query_payloads_private.json"
        private_payloads = [qi.payload.model_dump() for qi in private_incidents]
        private_payloads_path.write_text(json.dumps(private_payloads, indent=2))
        print(f"  Wrote {len(private_set.queries)} private-slice queries to {private_path}")

    # Generate multiple-choice remediation questions
    print("Building remediation questions...")
    remediation_qs = build_remediation_questions(query_incidents)
    rem_path = output / "remediation_questions.json"
    remediation_qs.save(rem_path)
    print(f"  Wrote {len(remediation_qs.questions)} remediation questions to {rem_path}")

    print("Done.")


def run():
    """Run baselines or a custom adapter against the corpus."""
    parser = argparse.ArgumentParser(
        description="Run Rhyme baselines or custom adapter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
tasks:
  1  Full-corpus retrieval — model sees all incidents, ranks by proximal cause
     Cost: ~75k tokens/query at 1000 incidents. Most expensive.
  2  Reasoning-only — model re-ranks BM25 top-20 candidates
     Cost: ~2k tokens/query. 30x cheaper than Task 1, often better results.
  3  Remediation — model picks best fix from 5 multiple-choice options
     Cost: ~500 tokens/query. Cheapest.

examples:
  rhyme-run --tasks 2,3                   # skip expensive full-corpus retrieval
  rhyme-run --tasks 1,2,3 --baseline bm25 # run everything with BM25
  rhyme-run --tasks 2 --adapter "python examples/anthropic_adapter.py"
""",
    )
    parser.add_argument(
        "--corpus", type=str, default="data/corpus.json",
        help="Path to corpus JSON",
    )
    parser.add_argument(
        "--queries", type=str, default="data/query_payloads.json",
        help="Path to query payloads JSON",
    )
    parser.add_argument(
        "--baseline", type=str, default="all",
        choices=["random", "bm25", "tfidf", "all"],
        help="Which baseline to run (default: all)",
    )
    parser.add_argument(
        "--adapter", type=str, default=None,
        help='Custom adapter command (e.g., "python my_model.py"). '
             "Uses subprocess protocol (JSON-lines on stdin/stdout). "
             "Overrides --baseline.",
    )
    parser.add_argument(
        "--tasks", type=str, default="1,2,3",
        help="Comma-separated list of tasks to run: 1,2,3 (default: all)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/results",
        help="Output directory for results",
    )
    parser.add_argument("-k", type=int, default=10, help="Top-k for retrieval")
    args = parser.parse_args()

    from .baselines import BM25Baseline, RandomBaseline, TfidfBaseline
    from .harness import Adapter, run_retrieval, run_reasoning_only, run_remediation
    from .models import Corpus, IncidentPayload, RemediationQuestionSet, RetrievalResult

    tasks = {int(t.strip()) for t in args.tasks.split(",")}

    corpus = Corpus.load(Path(args.corpus))

    raw_payloads = json.loads(Path(args.queries).read_text())
    query_payloads = [IncidentPayload.model_validate(p) for p in raw_payloads]

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    baselines: dict[str, Adapter] = {}

    if args.adapter:
        import shlex
        from .subprocess_adapter import SubprocessAdapter
        cmd = shlex.split(args.adapter)
        baselines["custom"] = SubprocessAdapter(cmd)
    else:
        if args.baseline in ("random", "all"):
            baselines["random"] = RandomBaseline()
        if args.baseline in ("bm25", "all"):
            baselines["bm25"] = BM25Baseline()
        if args.baseline in ("tfidf", "all"):
            baselines["tfidf"] = TfidfBaseline()

    # Load remediation questions if available
    rem_questions = None
    rem_path = Path(args.corpus).parent / "remediation_questions.json"
    if rem_path.exists():
        rem_questions = RemediationQuestionSet.load(rem_path)

    # Cost estimate for LLM adapters
    if args.adapter:
        n_queries = len(query_payloads)
        n_incidents = len(corpus.incidents)
        avg_summary_chars = sum(len(inc.payload.summary) for inc in corpus.incidents) / n_incidents
        est_corpus_tokens = int(n_incidents * avg_summary_chars / 3.5)  # ~3.5 chars per token

        print(f"Corpus: {n_incidents} incidents, {n_queries} queries")
        print(f"Tasks: {sorted(tasks)}")
        print()
        if 1 in tasks:
            est_t1 = est_corpus_tokens + 500  # corpus + prompt overhead
            print(f"  Task 1 (full corpus):   ~{est_t1:,} tokens/query x {n_queries} = ~{est_t1 * n_queries:,} tokens total")
        if 2 in tasks:
            est_t2 = int(est_corpus_tokens * 20 / n_incidents) + 500
            print(f"  Task 2 (reasoning-only): ~{est_t2:,} tokens/query x {n_queries} = ~{est_t2 * n_queries:,} tokens total")
        if 3 in tasks:
            est_t3 = 500
            print(f"  Task 3 (remediation):    ~{est_t3} tokens/query x {n_queries} = ~{est_t3 * n_queries:,} tokens total")
        total_est = sum([
            (est_corpus_tokens + 500) * n_queries if 1 in tasks else 0,
            (int(est_corpus_tokens * 20 / n_incidents) + 500) * n_queries if 2 in tasks else 0,
            500 * n_queries if 3 in tasks else 0,
        ])
        print(f"  Estimated total: ~{total_est:,} tokens")
        print()

    for name, adapter in baselines.items():
        # Task 1: full-corpus retrieval
        results = None
        if 1 in tasks:
            print(f"Running {name} (Task 1: full corpus)...")
            results = run_retrieval(adapter, query_payloads, corpus, k=args.k)
            out_path = output / f"{name}_results.json"
            out_path.write_text(json.dumps([r.model_dump() for r in results], indent=2))
            print(f"  Wrote {len(results)} results to {out_path}")

        # Task 2: reasoning-only (re-rank BM25 top-20)
        if 2 in tasks:
            print(f"Running {name} (Task 2: reasoning-only)...")
            results_t2 = run_reasoning_only(adapter, query_payloads, corpus, k=args.k)
            out_path_t2 = output / f"{name}_task2_results.json"
            out_path_t2.write_text(json.dumps([r.model_dump() for r in results_t2], indent=2))
            print(f"  Wrote {len(results_t2)} results to {out_path_t2}")

        # Task 3: remediation
        if 3 in tasks and rem_questions:
            # Task 3 needs Task 1 results for top matches. If Task 1 wasn't run, use BM25.
            if results is None:
                print(f"  Running BM25 pre-retrieval for Task 3 context...")
                bm25_for_context = BM25Baseline()
                results = run_retrieval(bm25_for_context, query_payloads, corpus, k=args.k)

            print(f"Running {name} (Task 3: remediation)...")
            rem_results = run_remediation(
                adapter, query_payloads, results, corpus, rem_questions,
            )
            if rem_results:
                out_path_rem = output / f"{name}_remediation_results.json"
                out_path_rem.write_text(json.dumps([r.model_dump() for r in rem_results], indent=2))
                print(f"  Wrote {len(rem_results)} remediation results to {out_path_rem}")

    print("Done.")


def score():
    """Score baseline results."""
    parser = argparse.ArgumentParser(description="Score Rhyme results")
    parser.add_argument(
        "--corpus", type=str, default="data/corpus.json",
        help="Path to corpus JSON",
    )
    parser.add_argument(
        "--queries", type=str, default="data/queries.json",
        help="Path to query set JSON (with ground truth)",
    )
    parser.add_argument(
        "--results", type=str, required=True,
        help="Path to retrieval results JSON",
    )
    parser.add_argument(
        "--remediation-results", type=str, default=None,
        help="Path to remediation results JSON (optional)",
    )
    parser.add_argument(
        "--remediation-questions", type=str, default="data/remediation_questions.json",
        help="Path to remediation questions JSON",
    )
    parser.add_argument("-k", type=int, default=10, help="Evaluation depth")
    args = parser.parse_args()

    from .models import Corpus, QuerySet, RemediationQuestionSet, RemediationResult, RetrievalResult
    from .scorer import score as do_score

    corpus = Corpus.load(Path(args.corpus))
    query_set = QuerySet.load(Path(args.queries))

    raw_results = json.loads(Path(args.results).read_text())
    results = [RetrievalResult.model_validate(r) for r in raw_results]

    rem_results = None
    rem_questions = None
    if args.remediation_results:
        raw_rem = json.loads(Path(args.remediation_results).read_text())
        rem_results = [RemediationResult.model_validate(r) for r in raw_rem]
        rem_questions = RemediationQuestionSet.load(Path(args.remediation_questions))

    report = do_score(
        results, query_set, corpus, k=args.k,
        remediation_results=rem_results,
        remediation_questions=rem_questions,
    )
    print(report.summary())


def probe():
    """Run the adversarial style probe on a corpus."""
    parser = argparse.ArgumentParser(description="Run adversarial style probe")
    parser.add_argument(
        "--corpus", type=str, default="data/corpus.json",
        help="Path to corpus JSON",
    )
    parser.add_argument("--folds", type=int, default=5, help="Cross-validation folds")
    args = parser.parse_args()

    from .models import Corpus
    from .style_probe import run_style_probe, print_probe_report

    corpus = Corpus.load(Path(args.corpus))
    results = run_style_probe(corpus, cv_folds=args.folds)
    print_probe_report(results)
