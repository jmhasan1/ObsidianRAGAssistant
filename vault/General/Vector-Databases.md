---
tags: [ai, infrastructure]
created: 2026-06-02
---

# Vector Databases

A vector database stores high-dimensional embeddings and supports fast
approximate nearest-neighbor (ANN) search, which is the backbone of
[[Retrieval-Augmented-Generation]].

## Popular Options

- **Chroma** — lightweight, file-based, great for local/small projects.
- **FAISS** — a library (not a server) from Meta, very fast, no persistence
  layer out of the box.
- **Pinecone / Weaviate / Qdrant** — managed or self-hosted servers with
  filtering, hybrid search, and horizontal scaling.

## Indexing Strategies

Most use HNSW (Hierarchical Navigable Small World) graphs, trading a small
amount of recall for large speed gains over brute-force search.

For small vaults (a few hundred to a few thousand notes), brute-force cosine
similarity is often fast enough that an ANN index is unnecessary overhead.

See also: [[Sentence-Embeddings]], [[Hybrid-Search]]
