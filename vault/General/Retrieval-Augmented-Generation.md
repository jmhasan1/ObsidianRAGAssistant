---
tags: [ai, rag, llm]
created: 2026-06-01
---

# Retrieval-Augmented Generation (RAG)

RAG is a technique that combines a retrieval system with a language model to
ground generated answers in external documents rather than relying purely on
the model's parametric memory.

## Core Pipeline

1. **Ingest** documents and split them into chunks.
2. **Embed** each chunk into a vector using a model like [[Sentence-Embeddings]].
3. **Store** vectors in a [[Vector-Databases]] for fast similarity search.
4. **Retrieve** the top-k relevant chunks for a user query.
5. **Generate** an answer conditioned on the retrieved chunks using an LLM.

## Why RAG Matters

- Reduces hallucination by grounding answers in real source text.
- Lets a model answer questions about private/recent data it was never
  trained on.
- Cheaper than fine-tuning for most knowledge-injection use cases.

See also: [[Hybrid-Search]], [[Prompt-Engineering-Basics]], [[DocuMind-Project]]
