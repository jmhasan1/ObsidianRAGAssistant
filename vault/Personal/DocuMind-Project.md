---
tags: [project, personal, rag]
created: 2026-07-10
---

# DocuMind Project

DocuMind is my main portfolio project — a local-first agentic RAG system.
The goal is to demonstrate production-grade RAG engineering, not just a
toy demo.

## Current State (v0.3, drafted)

- Hybrid retrieval: BM25 (sparse) + dense embeddings, combined with
  Reciprocal Rank Fusion — see [[Hybrid-Search]].
- Per-page PDF ingestion using PyMuPDF, including table detection.
- A four-mode benchmarking eval suite to compare retrieval configurations
  head-to-head.

## Why "Local-First"

Runs entirely on my own hardware (i5 10th gen, 16GB RAM, GTX 1650 Ti — see
[[Hardware-Constraints]]) without depending on paid cloud infra for the core
pipeline, which forces good engineering discipline around efficiency.

## Roadmap

- Merge the v0.3 hybrid-retrieval branch into the main repo.
- Move toward an agentic RAG design (see [[Agentic-AI-Concepts]]) where the
  system decides when and what to retrieve rather than always retrieving.

See also: [[Obsidian-RAG-Assistant-Project]], [[MLOps-Fundamentals]]
