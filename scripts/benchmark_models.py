#!/usr/bin/env python3
"""
Run Rhyme benchmark across multiple models and produce a comparison table.

Uses OpenRouter (openrouter.ai) by default — one API key for access to
OpenAI, Anthropic, Google, Mistral, Qwen, Meta, and more.

Usage:
  export OPENROUTER_API_KEY=sk-or-...
  docker build -t rhyme .
  docker run --rm -v $(pwd)/data:/app/data --entrypoint python rhyme \
    scripts/benchmark_models.py --tasks 2,3
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# Default model list — covers major providers via OpenRouter
DEFAULT_MODELS = [
    # OpenAI
    {"name": "GPT-4o", "model": "openai/gpt-4o"},
    {"name": "GPT-4o mini", "model": "openai/gpt-4o-mini"},
    {"name": "GPT-4.1 nano", "model": "openai/gpt-4.1-nano"},
    # Anthropic
    {"name": "Claude Sonnet 4", "model": "anthropic/claude-sonnet-4"},
    {"name": "Claude Haiku 3.5", "model": "anthropic/claude-3.5-haiku"},
    # Google
    {"name": "Gemini 2.0 Flash", "model": "google/gemini-2.0-flash-001"},
    {"name": "Gemini 2.5 Flash", "model": "google/gemini-2.5-flash-preview"},
    # Mistral
    {"name": "Mistral Small", "model": "mistralai/mistral-small-latest"},
    # Qwen
    {"name": "Qwen3 8B", "model": "qwen/qwen3-8b"},
    {"name": "Qwen3 32B", "model": "qwen/qwen3-32b"},
    # Meta
    {"name": "Llama 4 Scout", "model": "meta-llama/llama-4-scout"},
    {"name": "Llama 4 Maverick", "model": "meta-llama/llama-4-maverick"},
    # DeepSeek
    {"name": "DeepSeek V3", "model": "deepseek/deepseek-chat-v3-0324"},
]

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def run_model(model_config: dict, data_dir: str, tasks: str) -> dict | None:
    """Run a single model and return scored results."""
    name = model_config["name"]
    model_id = model_config["model"]
    safe_name = name.lower().replace(" ", "_").replace(".", "_")
    base_url = model_config.get("base_url", OPENROUTER_BASE_URL)
    api_key_env = model_config.get("api_key_env", "OPENROUTER_API_KEY")
    adapter_type = model_config.get("adapter", "openai_compat")

    api_key = os.environ.get(api_key_env, "")
    if not api_key:
        print(f"  SKIP {name}: {api_key_env} not set")
        return None

    output_dir = Path(data_dir) / f"bench_{safe_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set env for the adapter subprocess (inherits parent env)
    if adapter_type == "anthropic":
        os.environ["ANTHROPIC_API_KEY"] = api_key
        adapter_script = "examples/anthropic_adapter.py"
    else:
        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["OPENAI_BASE_URL"] = base_url
        adapter_script = "examples/openai_compat_adapter.py"
    os.environ["RHYME_MODEL"] = model_id

    # Find the project root (where examples/ lives)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    # Run rhyme via subprocess (calls the installed CLI)
    cmd = [
        sys.executable, "-m", "rhyme_bench.cli",
    ]
    # Actually, call the run function directly to avoid CLI discovery issues
    print(f"  Running {name} ({model_id})...")

    try:
        from rhyme_bench.baselines import BM25Baseline
        from rhyme_bench.harness import run_retrieval, run_reasoning_only, run_remediation
        from rhyme_bench.models import Corpus, IncidentPayload, RemediationQuestionSet

        corpus = Corpus.load(Path(data_dir) / "corpus.json")
        raw_payloads = json.loads((Path(data_dir) / "query_payloads.json").read_text())
        query_payloads = [IncidentPayload.model_validate(p) for p in raw_payloads]

        rem_questions = None
        rem_path = Path(data_dir) / "remediation_questions.json"
        if rem_path.exists():
            rem_questions = RemediationQuestionSet.load(rem_path)

        # Create adapter via subprocess
        from rhyme_bench.subprocess_adapter import SubprocessAdapter
        import shlex
        adapter_path = str(project_root / adapter_script)
        adapter = SubprocessAdapter([sys.executable, adapter_path])

        task_list = {int(t.strip()) for t in tasks.split(",")}
        results_data = {}

        # Task 1
        t1_results = None
        if 1 in task_list:
            t1_results = run_retrieval(adapter, query_payloads, corpus, k=10)
            out = output_dir / "custom_results.json"
            out.write_text(json.dumps([r.model_dump() for r in t1_results], indent=2))

        # Task 2
        t2_results = None
        if 2 in task_list:
            t2_results = run_reasoning_only(adapter, query_payloads, corpus, k=10)
            out = output_dir / "custom_task2_results.json"
            out.write_text(json.dumps([r.model_dump() for r in t2_results], indent=2))

        # Task 3
        if 3 in task_list and rem_questions:
            context_results = t1_results or t2_results
            if context_results is None:
                bm25 = BM25Baseline()
                context_results = run_retrieval(bm25, query_payloads, corpus, k=10)
            rem_results = run_remediation(adapter, query_payloads, context_results, corpus, rem_questions)
            if rem_results:
                out = output_dir / "custom_remediation_results.json"
                out.write_text(json.dumps([r.model_dump() for r in rem_results], indent=2))

        adapter.close()

        # Score
        from rhyme_bench.scorer import score
        from rhyme_bench.models import QuerySet

        qs = QuerySet.load(Path(data_dir) / "queries.json")

        if t2_results:
            report = score(t2_results, qs, corpus, k=10)
            eff = report.efficiency
            results_data["task2"] = {
                "precision": round(report.overall_precision_at_k_raw, 3),
                "retrieval": round(report.overall_retrieval_at_k, 3),
                "ece": round(report.overall_calibration_ece, 3),
                "tokens_per_query": round(eff.mean_tokens_per_query) if eff else 0,
            }

        if t1_results:
            report = score(t1_results, qs, corpus, k=10)
            eff = report.efficiency
            results_data["task1"] = {
                "precision": round(report.overall_precision_at_k_raw, 3),
                "retrieval": round(report.overall_retrieval_at_k, 3),
                "ece": round(report.overall_calibration_ece, 3),
                "tokens_per_query": round(eff.mean_tokens_per_query) if eff else 0,
            }

        rem_path = output_dir / "custom_remediation_results.json"
        if rem_path.exists() and (t1_results or t2_results):
            from rhyme_bench.models import RemediationResult
            rem_raw = json.loads(rem_path.read_text())
            rem = [RemediationResult.model_validate(r) for r in rem_raw]
            context = t2_results or t1_results
            report = score(context, qs, corpus, k=10,
                          remediation_results=rem, remediation_questions=rem_questions)
            if report.remediation_grades:
                g = report.remediation_grades.as_dict()
                results_data["remediation"] = {
                    "fixed": round(g["fixed"], 3),
                    "would_worsen": round(g["would_worsen"], 3),
                }

        print(f"  Done {name}: {results_data}")
        return {"name": name, "model": model_id, **results_data}

    except Exception as e:
        print(f"  ERROR {name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def print_table(all_results: list[dict], tasks: str):
    """Print a markdown comparison table."""
    task_list = [t.strip() for t in tasks.split(",")]

    print()
    print("## Rhyme Scores")
    print()

    if "2" in task_list:
        results_with_t2 = [r for r in all_results if "task2" in r]
        if results_with_t2:
            print("### Task 2: Reasoning-only (re-rank BM25 top-20)")
            print()
            print("| Model | Precision@10 | Retrieval@10 | ECE | Tokens/query |")
            print("|-------|-------------|-------------|-----|-------------|")
            for r in sorted(results_with_t2, key=lambda x: x["task2"]["precision"], reverse=True):
                t2 = r["task2"]
                print(f"| {r['name']} | {t2['precision']:.3f} | {t2['retrieval']:.3f} | {t2['ece']:.3f} | {t2['tokens_per_query']:,.0f} |")
            print()

    if "3" in task_list:
        results_with_rem = [r for r in all_results if "remediation" in r]
        if results_with_rem:
            print("### Task 3: Remediation (multiple-choice)")
            print()
            print("| Model | Correct fix | Would worsen |")
            print("|-------|------------|-------------|")
            for r in sorted(results_with_rem, key=lambda x: x["remediation"]["fixed"], reverse=True):
                rem = r["remediation"]
                print(f"| {r['name']} | {rem['fixed']:.0%} | {rem['would_worsen']:.0%} |")
            print()

    if "1" in task_list:
        results_with_t1 = [r for r in all_results if "task1" in r]
        if results_with_t1:
            print("### Task 1: Full-corpus retrieval")
            print()
            print("| Model | Precision@10 | Retrieval@10 | ECE | Tokens/query |")
            print("|-------|-------------|-------------|-----|-------------|")
            for r in sorted(results_with_t1, key=lambda x: x["task1"]["precision"], reverse=True):
                t1 = r["task1"]
                print(f"| {r['name']} | {t1['precision']:.3f} | {t1['retrieval']:.3f} | {t1['ece']:.3f} | {t1['tokens_per_query']:,.0f} |")
            print()


def main():
    parser = argparse.ArgumentParser(description="Benchmark multiple models with Rhyme")
    parser.add_argument("--data-dir", default="data", help="Data directory with corpus and queries")
    parser.add_argument("--tasks", default="2,3", help="Tasks to run (default: 2,3)")
    parser.add_argument("--config", default=None, help="Model config JSON file")
    parser.add_argument("--models", default=None, help="Comma-separated model names to filter")
    parser.add_argument("--output-json", default=None, help="Save raw results to JSON file")
    args = parser.parse_args()

    if args.config:
        models = json.loads(Path(args.config).read_text())
    else:
        models = DEFAULT_MODELS

    if args.models:
        requested = {m.strip().lower() for m in args.models.split(",")}
        models = [m for m in models if m["name"].lower() in requested or m["model"].split("/")[-1].lower() in requested]

    if not models:
        print("No models to run.")
        sys.exit(1)

    print(f"Benchmarking {len(models)} models on tasks {args.tasks}")
    print()

    all_results = []
    for model in models:
        result = run_model(model, args.data_dir, args.tasks)
        if result:
            all_results.append(result)

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(all_results, indent=2))
        print(f"\nRaw results saved to {args.output_json}")

    print_table(all_results, args.tasks)


if __name__ == "__main__":
    main()
