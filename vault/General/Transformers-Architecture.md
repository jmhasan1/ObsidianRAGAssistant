---
tags: [ai, nlp, deep-learning]
created: 2026-06-08
---

# Transformer Architecture

Transformers process sequences using self-attention instead of recurrence,
letting every token attend to every other token in parallel.

## Key Pieces

- **Self-attention** — computes a weighted combination of all tokens for
  each position, weights derived from query/key/value projections.
- **Multi-head attention** — runs several attention operations in parallel
  subspaces, then concatenates, letting the model capture different types
  of relationships simultaneously.
- **Positional encoding** — since attention has no inherent notion of order,
  position information is injected separately.

## Why This Matters for LLMs and RAG

Every embedding model ([[Sentence-Embeddings]]) and generation model used in
a RAG pipeline ([[Retrieval-Augmented-Generation]]) is a transformer under
the hood — understanding attention helps explain why context length,
chunk size, and prompt structure all affect output quality.

See also: [[Sentence-Embeddings]], [[Prompt-Engineering-Basics]]
