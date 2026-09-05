---
tags: [ai, mlops]
created: 2026-06-07
---

# MLOps Fundamentals

MLOps applies DevOps discipline (versioning, CI/CD, monitoring) to machine
learning systems, where the "artifact" being shipped includes both code and
data/models.

## Key Practices

- **Experiment tracking** — logging hyperparameters, metrics, and configs
  for every run so results are reproducible.
- **Evaluation suites** — automated benchmarks that catch regressions before
  deployment (relevant for RAG: retrieval accuracy, answer faithfulness).
- **Versioning data and models**, not just code.
- **Monitoring in production** — tracking drift, latency, and failure modes
  after deployment, not just at training time.

## For RAG Systems Specifically

Evaluation is trickier than classic ML because "correctness" is fuzzy. Useful
signals include retrieval hit-rate (did the right chunk get retrieved?) and
faithfulness (does the generated answer only use the retrieved context?).

See also: [[Agentic-AI-Concepts]], [[DocuMind-Project]]
