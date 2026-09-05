---
tags: [ai, rag]
created: 2026-06-03
---

# Chunking Strategies

How you split documents before embedding often matters more than which
embedding model you choose.

## Approaches

- **Fixed-size** — split every N tokens with some overlap. Simple, works
  everywhere, but can cut sentences/ideas in half.
- **Heading-based** (best for Markdown/Obsidian) — split at `#`/`##`
  boundaries so each chunk is a coherent section.
- **Semantic chunking** — use an embedding model to detect topic shifts and
  split there. More expensive, marginal gains for small vaults.

## Overlap

A 10-20% overlap between adjacent chunks helps preserve context that spans a
chunk boundary, at the cost of some duplicate retrieval.

## For Obsidian Vaults Specifically

Notes are usually already short and single-topic, so a reasonable default is
"one chunk per note, or one chunk per heading section if the note is long."

See also: [[Retrieval-Augmented-Generation]], [[DocuMind-Project]]
