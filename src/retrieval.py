"""Retrieval utilities for dense, lexical, and hybrid search."""

from __future__ import annotations

import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+(?:-[A-Za-z0-9_]+)*")


def tokenize(text: str) -> list[str]:
    """Normalize text into simple lexical-search tokens."""
    return TOKEN_RE.findall(text.lower())


class BM25:
    """Small self-contained BM25 index suitable for a small local vault."""

    def __init__(
        self,
        documents: list[str],
        k1: float = 1.5,
        b: float = 0.75,
    ):
        if not documents:
            raise ValueError("BM25 requires at least one document.")

        self.documents = documents
        self.k1 = k1
        self.b = b

        self.tokenized_documents = [tokenize(doc) for doc in documents]
        self.document_lengths = [
            len(tokens) for tokens in self.tokenized_documents
        ]
        self.avg_document_length = sum(self.document_lengths) / len(
            self.document_lengths
        )

        self.document_frequency: Counter[str] = Counter()

        for tokens in self.tokenized_documents:
            self.document_frequency.update(set(tokens))

        self.num_documents = len(documents)

    def score(self, query: str) -> list[float]:
        """Return a BM25 score for every indexed document."""
        query_tokens = tokenize(query)

        if not query_tokens:
            return [0.0] * self.num_documents

        query_terms = set(query_tokens)
        scores = [0.0] * self.num_documents

        for index, tokens in enumerate(self.tokenized_documents):
            term_counts = Counter(tokens)
            document_length = self.document_lengths[index]

            score = 0.0

            for term in query_terms:
                term_frequency = term_counts.get(term, 0)

                if term_frequency == 0:
                    continue

                document_frequency = self.document_frequency.get(term, 0)

                idf = math.log(
                    1
                    + (
                        self.num_documents
                        - document_frequency
                        + 0.5
                    )
                    / (document_frequency + 0.5)
                )

                denominator = term_frequency + self.k1 * (
                    1
                    - self.b
                    + self.b
                    * document_length
                    / self.avg_document_length
                )

                score += (
                    idf
                    * term_frequency
                    * (self.k1 + 1)
                    / denominator
                )

            scores[index] = score

        return scores

    def rank(self, query: str) -> list[int]:
        """Return document indexes ranked from most to least relevant."""
        scores = self.score(query)
        return sorted(
            range(self.num_documents),
            key=lambda index: scores[index],
            reverse=True,
        )


def reciprocal_rank_fusion(
    rankings: list[list[int]],
    k: int = 60,
) -> dict[int, float]:
    """Fuse multiple ranked lists using Reciprocal Rank Fusion."""
    fused_scores: dict[int, float] = {}

    for ranking in rankings:
        for rank, document_index in enumerate(ranking, start=1):
            fused_scores[document_index] = fused_scores.get(
                document_index,
                0.0,
            ) + 1.0 / (k + rank)

    return fused_scores


def rank_fused_results(
    rankings: list[list[int]],
    limit: int | None = None,
) -> list[tuple[int, float]]:
    """Return document indexes and RRF scores in descending order."""
    fused_scores = reciprocal_rank_fusion(rankings)

    results = sorted(
        fused_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    if limit is not None:
        results = results[:limit]

    return results

DEFAULT_DENSE_RELEVANCE_THRESHOLD = 0.35
DEFAULT_BM25_RELEVANCE_THRESHOLD = 6.0
DEFAULT_MIN_DENSE_FOR_LEXICAL_MATCH = 0.25


def is_relevant(
    results: list[dict],
    dense_threshold: float = DEFAULT_DENSE_RELEVANCE_THRESHOLD,
    bm25_threshold: float = DEFAULT_BM25_RELEVANCE_THRESHOLD,
    min_dense_for_lexical_match: float = DEFAULT_MIN_DENSE_FOR_LEXICAL_MATCH,
) -> bool:
    """Return whether the top retrieved result contains sufficient evidence."""
    if not results:
        return False

    top_result = results[0]

    dense_score = float(top_result.get("dense_score", 0.0))
    bm25_score = float(top_result.get("bm25_score", 0.0))

    strong_semantic_match = dense_score >= dense_threshold
    strong_lexical_match = (
        bm25_score >= bm25_threshold
        and dense_score >= min_dense_for_lexical_match
    )

    return strong_semantic_match or strong_lexical_match