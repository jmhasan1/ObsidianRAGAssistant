---
tags: [project, personal, rag]
created: 2026-09-04
---

# Obsidian RAG Assistant Project

This is the MVP I'm building for The Skillians' Generative AI Developer
Intern Build Sprint (submission deadline 07 September 2026).

## Goal

Let a user point the assistant at an Obsidian vault and ask natural-language
questions, getting answers grounded in the actual notes with citations back
to source files.

## v1 Scope

- Dense-only retrieval (MiniLM embeddings + Chroma) — hybrid search
  ([[Hybrid-Search]]) is deliberately deferred to keep the build shippable
  in a 5-day window.
- Generation via Groq's free-tier API (fast open-weight models), with an
  OpenAI key as a low-cost fallback.
- Streamlit UI, deployed on Streamlit Community Cloud for a public demo
  link.
- This very vault — a mix of general AI/ML notes and personal project notes
  — is the demo dataset, chosen so retrieval quality is easy to sanity-check
  by hand.

## Relationship to DocuMind

Reuses architectural lessons from [[DocuMind-Project]] (chunking, retrieval
design) but intentionally simpler — no hybrid search, no custom eval suite —
since the deliverable here is a working MVP, not a research benchmark.

See also: [[Retrieval-Augmented-Generation]], [[Job-Search-Strategy]]
