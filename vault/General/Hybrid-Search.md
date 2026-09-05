---
tags: [ai, rag, retrieval]
created: 2026-06-04
---

# Hybrid Search

Hybrid search combines **sparse** retrieval (keyword-based, e.g. BM25) with
**dense** retrieval (embedding similarity) to get the strengths of both:
exact term matching plus semantic matching.

## Combining Scores

A common technique is **Reciprocal Rank Fusion (RRF)**, which merges two
ranked lists by rank position rather than raw score, avoiding the need to
normalize incomparable similarity/BM25 scores.

## When It Helps Most

- Queries with specific keywords, acronyms, or names that embeddings alone
  might blur together.
- Small corpora where dense retrieval alone sometimes returns near-ties that
  keyword overlap can disambiguate.

This is a planned upgrade path for the assistant described in
[[Obsidian-RAG-Assistant-Project]] — the v1 build uses dense-only retrieval
to ship faster, with hybrid search as a documented next step.

See also: [[Retrieval-Augmented-Generation]], [[DocuMind-Project]]
