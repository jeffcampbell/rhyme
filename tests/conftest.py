"""Shared fixtures for Sifter tests."""

import pytest
from pathlib import Path

from sifter_bench.generator import generate_corpus
from sifter_bench.queries import build_query_set, build_remediation_questions
from sifter_bench.taxonomy import CauseClass


@pytest.fixture(scope="session")
def small_corpus():
    """A small corpus (5 per class = 100 incidents) for fast tests."""
    return generate_corpus(incidents_per_class=5, seed=99)


@pytest.fixture(scope="session")
def query_set_and_incidents(small_corpus):
    """Query set + query incidents from the small corpus."""
    qs, qi = build_query_set(small_corpus, total_queries=20, seed=99)
    return qs, qi


@pytest.fixture(scope="session")
def remediation_questions(query_set_and_incidents):
    _, qi = query_set_and_incidents
    return build_remediation_questions(qi)


@pytest.fixture
def prose_pools_dir(tmp_path):
    """An empty prose pools directory (tests run without LLM prose pools)."""
    d = tmp_path / "prose_pools"
    d.mkdir()
    return d
