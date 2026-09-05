---
tags: [personal, hardware]
created: 2026-07-11
---

# Hardware Constraints

My local machine: i5 10th Gen CPU, 16GB RAM, GTX 1650 Ti (4GB VRAM).

## What This Rules Out

- Running large local LLMs for real-time inference — the 1650 Ti's 4GB VRAM
  is too small for anything beyond small quantized models at usable speed.
- Training or fine-tuning anything beyond small models.

## How It Shapes My Choices

- Prefer CPU-friendly embedding models (e.g. MiniLM) over large local
  generation models.
- Use hosted/API LLMs (Groq, OpenAI, Anthropic) for the generation step in
  projects like [[DocuMind-Project]] and [[Obsidian-RAG-Assistant-Project]]
  rather than trying to self-host generation.
- Favor lightweight, file-based vector stores (Chroma) over heavier
  server-based ones for local development.

See also: [[DocuMind-Project]]
