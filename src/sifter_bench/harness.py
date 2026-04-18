"""
Evaluation harness: adapter interface and query runner.

Models implement the Adapter protocol. The harness issues queries against
the corpus and collects structured outputs for scoring.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .models import (
    Corpus,
    IncidentPayload,
    RankedMatch,
    RemediationChoice,
    RemediationQuestion,
    RemediationQuestionSet,
    RemediationResult,
    RetrievalResult,
    TokenUsage,
)


class RetrieveOutput:
    """Return type for Adapter.retrieve() — matches plus optional token usage."""

    __slots__ = ("matches", "token_usage")

    def __init__(
        self,
        matches: list[RankedMatch],
        token_usage: TokenUsage | None = None,
    ):
        self.matches = matches
        self.token_usage = token_usage


class Adapter(ABC):
    """Interface that model implementations must satisfy."""

    @abstractmethod
    def retrieve(
        self,
        query: IncidentPayload,
        corpus: list[IncidentPayload],
        k: int = 10,
    ) -> list[RankedMatch] | RetrieveOutput:
        """Return ranked matches for the query from the corpus.

        Args:
            query: The incident to find matches for.
            corpus: All candidate incidents (payloads only, no labels).
            k: Maximum number of matches to return.

        Returns:
            Ranked list of matches with confidence scores [0, 1],
            or a RetrieveOutput wrapping matches + token usage.
        """
        ...

    def remediate(
        self,
        query: IncidentPayload,
        top_matches: list[IncidentPayload],
        choices: list[RemediationChoice],
    ) -> str | None:
        """Select a remediation from the given multiple-choice options.

        Args:
            query: The incident to remediate.
            top_matches: Top-3 similar incidents from retrieval.
            choices: 5 multiple-choice options labeled A-E.

        Returns:
            The selected label (A-E), or None to skip Task 3.
        """
        return None


def run_retrieval(
    adapter: Adapter,
    query_incidents: list[IncidentPayload],
    corpus: Corpus,
    k: int = 10,
) -> list[RetrievalResult]:
    """Run Task 1: full-corpus retrieval for each query.

    Removes the query itself from the corpus before presenting to the adapter.
    """
    results = []
    all_payloads = corpus.payloads()

    for query in query_incidents:
        # Exclude the query incident from the candidate pool
        candidates = [p for p in all_payloads if p.incident_id != query.incident_id]
        raw = adapter.retrieve(query, candidates, k=k)

        # Handle both return types
        if isinstance(raw, RetrieveOutput):
            matches = raw.matches[:k]
            token_usage = raw.token_usage
        else:
            matches = raw[:k]
            token_usage = None

        results.append(RetrievalResult(
            query_id=query.incident_id,
            ranked_matches=matches,
            token_usage=token_usage,
        ))

    return results


def run_reasoning_only(
    adapter: Adapter,
    query_incidents: list[IncidentPayload],
    corpus: Corpus,
    pre_retrieve_k: int = 20,
    k: int = 10,
    pre_retriever: Adapter | None = None,
) -> list[RetrievalResult]:
    """Run Task 2: reasoning-only retrieval.

    Uses a pre-retriever to narrow the corpus to top-`pre_retrieve_k`
    candidates, then asks the adapter to re-rank from that smaller set.
    Isolates model reasoning quality from retrieval quality.

    Args:
        pre_retriever: Adapter used for initial narrowing. Defaults to BM25Baseline.
    """
    if pre_retriever is None:
        from .baselines import BM25Baseline
        pre_retriever = BM25Baseline()

    results = []
    all_payloads = corpus.payloads()

    for query in query_incidents:
        candidates = [p for p in all_payloads if p.incident_id != query.incident_id]

        # Pre-retrieve top-N
        pre_raw = pre_retriever.retrieve(query, candidates, k=pre_retrieve_k)
        pre_matches = pre_raw.matches if isinstance(pre_raw, RetrieveOutput) else pre_raw
        pre_ids = {m.incident_id for m in pre_matches}
        narrowed = [p for p in candidates if p.incident_id in pre_ids]

        # Let the adapter re-rank from the narrowed set
        raw = adapter.retrieve(query, narrowed, k=k)

        if isinstance(raw, RetrieveOutput):
            matches = raw.matches[:k]
            token_usage = raw.token_usage
        else:
            matches = raw[:k]
            token_usage = None

        results.append(RetrievalResult(
            query_id=query.incident_id,
            ranked_matches=matches,
            token_usage=token_usage,
        ))

    return results


def run_remediation(
    adapter: Adapter,
    query_incidents: list[IncidentPayload],
    retrieval_results: list[RetrievalResult],
    corpus: Corpus,
    question_set: RemediationQuestionSet,
) -> list[RemediationResult]:
    """Run Task 3: multiple-choice remediation for each query."""
    payload_map = {p.incident_id: p for p in corpus.payloads()}
    question_map = {q.query_id: q for q in question_set.questions}
    results = []

    for query, retrieval in zip(query_incidents, retrieval_results):
        question = question_map.get(query.incident_id)
        if question is None:
            continue

        top_payloads = [
            payload_map[m.incident_id]
            for m in retrieval.ranked_matches[:3]
            if m.incident_id in payload_map
        ]
        selected = adapter.remediate(query, top_payloads, question.choices)
        if selected is not None:
            results.append(RemediationResult(
                query_id=query.incident_id,
                selected_label=selected,
            ))

    return results
