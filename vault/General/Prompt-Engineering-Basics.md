---
tags: [ai, prompting]
created: 2026-06-05
---

# Prompt Engineering Basics

Prompt engineering is the practice of structuring instructions and context to
reliably steer an LLM's output.

## Techniques That Generalize Well

- **Be explicit about format** — ask for JSON, bullet points, or a specific
  structure if downstream code parses the output.
- **Few-shot examples** — show 1-3 examples of desired input/output pairs.
- **Chain-of-thought** — asking the model to reason step-by-step before
  answering improves accuracy on multi-step problems.
- **Grounding instructions** — for RAG specifically, explicitly tell the
  model to only answer from provided context and to say so if the context
  doesn't contain the answer.

## RAG-Specific Prompting

A good RAG system prompt usually includes:
1. The retrieved chunks, clearly delimited.
2. An instruction to cite which chunk/source supports each claim.
3. An instruction to admit uncertainty rather than hallucinate when the
   retrieved context is insufficient.

See also: [[Retrieval-Augmented-Generation]], [[Agentic-AI-Concepts]]
