---
tags: [ai, nlp]
created: 2026-06-02
---

# Sentence Embeddings

Sentence embeddings map a piece of text to a dense vector such that
semantically similar text ends up close together in vector space.

## Common Models

- `all-MiniLM-L6-v2` — small (22M params), fast on CPU, good baseline quality.
- `bge-small` / `bge-base` — strong open-source retrieval-tuned embeddings.
- OpenAI `text-embedding-3-small` — hosted, no local compute needed.

## Practical Notes

- Always embed queries with the *same* model used to embed the documents.
- Normalize vectors (L2 norm) before cosine similarity for consistent scoring.
- Smaller models are usually fine for retrieval — embedding quality matters
  less than chunking quality in most RAG failures.

See also: [[Vector-Databases]], [[Chunking-Strategies]]
