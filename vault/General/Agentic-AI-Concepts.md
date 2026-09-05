---
tags: [ai, agents]
created: 2026-06-06
---

# Agentic AI Concepts

An "agent" wraps an LLM with the ability to take actions — calling tools,
querying data, executing code — and to loop on its own output until a task
is complete, rather than producing one static response.

## Core Components

- **Planner/reasoner** — decides what step to take next.
- **Tools** — functions the agent can call (search, retrieval, code exec,
  API calls).
- **Memory** — short-term (conversation) and long-term (retrieved knowledge,
  vector stores).
- **Loop control** — a stopping condition so the agent doesn't loop forever.

## Multi-Agent Systems

Splitting responsibilities across specialized agents (e.g. a "retriever
agent" and a "critic agent" that checks the retriever's output) can improve
reliability, at the cost of more orchestration complexity and latency.

## Relevance to RAG

A simple RAG pipeline is retrieve-then-generate. An "agentic RAG" system
instead lets the model decide *whether* to retrieve, *what* to search for,
and *whether the retrieved context is good enough* before answering — this
is the direction planned for [[DocuMind-Project]].

See also: [[Prompt-Engineering-Basics]], [[MLOps-Fundamentals]]
