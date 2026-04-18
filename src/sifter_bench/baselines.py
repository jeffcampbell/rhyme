"""
Baseline adapters shipped with the benchmark.

1. RandomBaseline — random retrieval (floor)
2. BM25Baseline — BM25 over full payload text
3. EmbeddingBaseline — TF-IDF cosine similarity (stand-in for embedding model)

LLM-based baselines (4, 5 from spec §8) are deferred to the full v1 build
since they require API keys and are not deterministic.
"""

from __future__ import annotations

import random

from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .harness import Adapter
from .models import IncidentPayload, RankedMatch, RemediationChoice


def _payload_to_text(payload: IncidentPayload) -> str:
    """Flatten an incident payload to a single text string for retrieval.

    Thin wrapper around IncidentPayload.to_text() — kept as a module-level
    function for backward compatibility with tests that import it directly.
    """
    return payload.to_text()


class RandomBaseline(Adapter):
    """Baseline 1: random retrieval + random remediation. Establishes the floor."""

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)

    def retrieve(
        self,
        query: IncidentPayload,
        corpus: list[IncidentPayload],
        k: int = 10,
    ) -> list[RankedMatch]:
        candidates = list(corpus)
        self._rng.shuffle(candidates)
        return [
            RankedMatch(
                incident_id=c.incident_id,
                confidence=round(self._rng.random(), 3),
            )
            for c in candidates[:k]
        ]

    def remediate(
        self,
        query: IncidentPayload,
        top_matches: list[IncidentPayload],
        choices: list[RemediationChoice],
    ) -> str | None:
        return self._rng.choice(choices).label if choices else None


class BM25Baseline(Adapter):
    """Baseline 2: BM25 over full payload text."""

    def retrieve(
        self,
        query: IncidentPayload,
        corpus: list[IncidentPayload],
        k: int = 10,
    ) -> list[RankedMatch]:
        corpus_texts = [_payload_to_text(p) for p in corpus]
        tokenized = [doc.lower().split() for doc in corpus_texts]
        bm25 = BM25Okapi(tokenized)

        query_text = _payload_to_text(query)
        scores = bm25.get_scores(query_text.lower().split())

        # Normalize scores to [0, 1]
        max_score = max(scores) if max(scores) > 0 else 1.0
        ranked_indices = sorted(range(len(scores)), key=lambda i: -scores[i])

        results = []
        for idx in ranked_indices[:k]:
            results.append(RankedMatch(
                incident_id=corpus[idx].incident_id,
                confidence=round(float(scores[idx] / max_score), 3),
            ))
        return results


class TfidfBaseline(Adapter):
    """Baseline 3: TF-IDF cosine similarity.

    Stand-in for the frozen embedding model baseline. Uses TF-IDF instead
    of a neural embedding model to avoid external dependencies.
    """

    def retrieve(
        self,
        query: IncidentPayload,
        corpus: list[IncidentPayload],
        k: int = 10,
    ) -> list[RankedMatch]:
        corpus_texts = [_payload_to_text(p) for p in corpus]
        query_text = _payload_to_text(query)

        vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
        corpus_vectors = vectorizer.fit_transform(corpus_texts)
        query_vector = vectorizer.transform([query_text])

        similarities = cosine_similarity(query_vector, corpus_vectors)[0]
        ranked_indices = sorted(range(len(similarities)), key=lambda i: -similarities[i])

        results = []
        for idx in ranked_indices[:k]:
            results.append(RankedMatch(
                incident_id=corpus[idx].incident_id,
                confidence=round(float(similarities[idx]), 3),
            ))
        return results
