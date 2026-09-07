# 3.7 Final Evaluation

## 1. Evaluation Objective

The evaluation measures whether the Obsidian Vault RAG Assistant can:

1. Retrieve relevant vault content for answerable questions.
2. Correctly identify questions that are unsupported by the vault.
3. Avoid unnecessary LLM generation for irrelevant questions.
4. Generate answers successfully when sufficient evidence is retrieved.
5. Include citations in generated answers.
6. Keep citations grounded in the notes retrieved for the query.

The evaluation is designed for the current MVP implementation and uses a small, labeled benchmark derived from the included demo vault.

---

## 2. Evaluation Dataset

The benchmark contains **12 queries**:

| Category | Queries |
|---|---:|
| Answerable | 8 |
| Unanswerable | 4 |
| **Total** | **12** |

### Answerable queries

Each answerable query has one or more expected source notes.

| ID | Question | Expected source |
|---|---|---|
| q01 | How does hybrid search combine dense and lexical retrieval? | Hybrid Search |
| q02 | How does Reciprocal Rank Fusion combine retrieval results? | Hybrid Search |
| q03 | What embedding model is used for the local retrieval baseline? | Sentence Embeddings |
| q04 | What hardware constraints are described for the local setup? | Hardware Constraints |
| q05 | Why is heading-based chunking a good fit for an Obsidian vault? | Chunking Strategies |
| q06 | What are the core components of an agentic AI system? | Agentic AI Concepts |
| q07 | What are the main stages of a RAG pipeline? | Retrieval-Augmented Generation (RAG) |
| q08 | Why is brute-force cosine similarity acceptable for a small vault? | Vector Databases |

### Unanswerable queries

These questions are intentionally outside the knowledge contained in the demo vault:

| ID | Question |
|---|---|
| q09 | What is the capital of France? |
| q10 | Who won the 2026 FIFA World Cup? |
| q11 | What is the current stock price of NVIDIA? |
| q12 | What is the weather forecast for Guwahati tomorrow? |

The unanswerable set tests whether the relevance gate prevents the system from answering questions using unsupported external or parametric knowledge.

---

## 3. Evaluation Methodology

The evaluation runs through the same retrieval and generation components used by the application.

For each query:

```text
Question
   │
   ▼
Hybrid retrieval
   │
   ├── Dense retrieval
   │
   └── BM25 retrieval
           │
           ▼
        RRF fusion
           │
           ▼
        Top-K results
           │
           ▼
      Relevance gate
        /        \
      reject     accept
        │           │
        ▼           ▼
     Abstain     LLM generation
                    │
                    ▼
                 Citation
```

The evaluator uses `TOP_K = 4`.

For answerable questions, retrieval is compared against the expected source notes.

For unanswerable questions, the expected behavior is rejection by the relevance gate.

For accepted queries, generation is attempted and the resulting answer is checked for citation coverage and citation validity.

---

## 4. Metrics

### Hit@4

Measures whether at least one expected source note appears in the top four retrieved results.

```text
Hit@4 = queries with a relevant source in top-4 / answerable queries
```

Higher is better.

### Mean Reciprocal Rank (MRR)

Measures how highly the first relevant source appears in the ranked results.

```text
RR = 1 / rank of first relevant result
MRR = mean(RR)
```

Higher is better.

### Relevance-gate accuracy

Measures whether the relevance gate correctly classifies both answerable and unanswerable queries.

A correct classification is:

- answerable query → accepted
- unanswerable query → rejected

### Answerable recall

Measures the proportion of answerable queries that are accepted by the relevance gate.

### Unanswerable rejection

Measures the proportion of unanswerable queries that are correctly rejected before generation.

### Generation success

Measures whether LLM generation successfully produces an answer for queries that pass the relevance gate.

### Citation coverage

Measures the proportion of successfully generated answers containing at least one citation.

### Citation validity

Measures whether the citations in generated answers refer to notes that were actually retrieved for that query.

Citations may contain section information, for example:

```text
(Note: Hybrid Search | Section: Combining Scores)
```

The evaluator extracts the note title:

```text
Hybrid Search
```

and checks that the note was present in the retrieved source set.

This provides note-level grounding validation while retaining section-level citation information in the generated answer.

---

## 5. Results

The final benchmark produced the following results:

| Metric | Result |
|---|---:|
| Evaluation queries | 12 |
| Answerable queries | 8 |
| Unanswerable queries | 4 |
| Hit@4 | 87.5% |
| MRR | 87.5% |
| Relevance-gate accuracy | 100% |
| Answerable recall | 100% |
| Unanswerable rejection | 100% |
| Generation success | 100% |
| Citation coverage | 100% |
| Citation validity | 100% |

Detailed per-query results are stored in:

```text
evaluation/results.json
```

---

## 6. Results Interpretation

### Retrieval

The system achieved:

```text
Hit@4 = 87.5%
MRR    = 87.5%
```

This means the expected source was retrieved within the top four results for **7 of the 8 answerable benchmark queries**.

One answerable query, **q07**, did not retrieve its expected `Retrieval-Augmented Generation (RAG)` note within the top four results.

The system intentionally reports this miss rather than tuning the benchmark or retrieval thresholds specifically to force a perfect retrieval score.

This provides a more realistic view of the current retrieval behavior.

### Relevance gating

The relevance gate achieved:

```text
Gate accuracy          = 100%
Answerable recall      = 100%
Unanswerable rejection = 100%
```

All eight answerable queries were allowed to proceed to generation, while all four unanswerable queries were rejected.

This demonstrates that the gate provides an additional protection layer between retrieval and LLM generation.

### Generation

All queries that passed the relevance gate generated successfully:

```text
Generation success = 100%
```

This indicates that the configured generation path was operational for the entire accepted benchmark.

### Citation grounding

All successful generated answers contained citations:

```text
Citation coverage = 100%
```

All evaluated citations referred to notes that were present in the retrieved source set:

```text
Citation validity = 100%
```

Citation validity is evaluated at the note level while section-level information is preserved in the citation string.

---

## 7. Known Retrieval Limitation

The current benchmark exposes one retrieval limitation.

For **q07**:

**Question:**

```text
What are the main stages of a RAG pipeline?
```

**Expected source:**

```text
Retrieval-Augmented Generation (RAG)
```

However, that source was not included in the top four retrieved results.

The relevance gate nevertheless accepted the query because the retrieved evidence passed the configured relevance thresholds.

This is an important distinction:

> A query can be classified as relevant while the ideal source is not ranked within the selected top-k results.

The current MVP therefore demonstrates useful hybrid retrieval and abstention behavior while leaving room for future retrieval improvements.

---

## 8. Reproducibility

The evaluation can be reproduced from the repository root with:

```bash
uv run python evaluation/evaluate.py
```

or:

```bash
uv run python -m evaluation.evaluate
```

The evaluation output is written to:

```text
evaluation/results.json
```

The automated test suite can be run with:

```bash
uv run pytest
```

The current suite contains **23 tests** covering retrieval, relevance gating, evaluation metrics, and citation evaluation.

Linting can be verified with:

```bash
uv run ruff check .
```

---

## 9. Evaluation Limitations

This evaluation is intentionally small and should not be interpreted as a production-scale benchmark.

Current limitations include:

- Only 12 labeled queries are evaluated.
- The benchmark is derived from a small demo vault.
- The answerable set covers a limited range of question types.
- Retrieval metrics use the expected source-note labels rather than human relevance judgments for every retrieved chunk.
- Generation quality is evaluated primarily through successful generation and citation grounding rather than a comprehensive semantic quality benchmark.
- No LLM-as-a-judge metric is included in the MVP evaluation.
- The benchmark does not measure latency, throughput, or API cost.

These limitations are appropriate for the scope of the one-week build sprint.