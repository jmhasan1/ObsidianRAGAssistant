---
tags: [ai, llm]
created: 2026-06-09
---

# Fine-Tuning vs Prompting vs RAG

Three different ways to adapt an LLM's behavior, often confused with each
other.

## Prompting

Cheapest and fastest to iterate on. Good for steering tone, format, and
reasoning style. Doesn't add new factual knowledge reliably.

## RAG

Best for injecting up-to-date or private knowledge the model wasn't trained
on, without touching model weights. See [[Retrieval-Augmented-Generation]].

## Fine-Tuning

Adjusts model weights on a custom dataset. Good for teaching a consistent
style, format, or domain-specific behavior at scale, but expensive, slower
to iterate, and a poor tool for keeping knowledge current (you'd have to
retrain every time facts change).

## Rule of Thumb

Start with prompting. Add RAG when the model needs facts it doesn't have.
Only fine-tune when neither solves a persistent behavioral gap.

See also: [[Prompt-Engineering-Basics]], [[MLOps-Fundamentals]]
