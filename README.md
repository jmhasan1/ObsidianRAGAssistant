# Obsidian Vault RAG Assistant

A Retrieval-Augmented Generation (RAG) assistant that answers natural-language questions about an Obsidian vault using the vault's actual Markdown notes as its knowledge source.

The system combines dense semantic retrieval with BM25 lexical retrieval, fuses the results using Reciprocal Rank Fusion (RRF), applies a relevance gate to avoid answering unsupported questions, and generates grounded answers with source citations.

Built as the MVP submission for **The Skillians' Generative AI Developer Intern Build Sprint**.

**Live Demo:** **Live Demo:** [Obsidian Vault RAG Assistant](https://obsidianragassistant.streamlit.app/)<br>
**GitHub:** https://github.com/jmhasan1/obsidian-rag-assistant

---

## What it does

The assistant turns a folder of Obsidian-style Markdown notes into a searchable knowledge base and answers questions using only retrieved vault content.

The pipeline:

1. **Ingests** Markdown notes from the vault.
2. **Parses** YAML frontmatter, headings, and Obsidian `[[wikilinks]]`.
3. **Chunks** notes at heading boundaries to preserve section-level context.
4. **Embeds** chunks using `all-MiniLM-L6-v2`.
5. **Retrieves** relevant chunks using both:
   - dense cosine-similarity search
   - BM25 lexical search
6. **Fuses** the two rankings using Reciprocal Rank Fusion (RRF).
7. **Applies a relevance gate** before generation to reject clearly unsupported questions.
8. **Generates** an answer using retrieved context only.
9. **Cites** the source note and section used for the answer.
10. **Displays** the result through a Streamlit chat interface.

The result is a lightweight RAG system designed specifically for a small Obsidian knowledge base without requiring a heavyweight vector database.

---

## Architecture

```text
Obsidian Vault
      │
      ▼
┌───────────────────┐
│    Ingestion      │
│                   │
│ YAML frontmatter  │
│ Headings          │
│ [[wikilinks]]     │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  Heading-based    │
│     Chunking      │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│     Sentence      │
│    Embeddings     │
│   MiniLM-L6-v2    │
└─────────┬─────────┘
          │
          ▼
┌───────────────────────────────┐
│       Hybrid Retrieval        │
│                               │
│  Dense cosine      BM25       │
│       │             │         │
│       └──────┬──────┘         │
│              ▼                │
│          RRF fusion            │
└─────────────┬─────────────────┘
              │
              ▼
┌───────────────────┐
│  Relevance Gate   │
│                   │
│ Relevant? ────────┼── No ──► Abstain
└─────────┬─────────┘
          │ Yes
          ▼
┌───────────────────┐
│   Grounded LLM    │
│    Generation     │
│                   │
│  Groq primary     │
│  OpenAI fallback  │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│   Cited Answer    │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Streamlit Chat UI │
└───────────────────┘
```

## Tech stack

| Component | Implementation | Why |
|---|---|---|
| Language | Python 3.12 | Simple ecosystem for RAG experimentation and deployment |
| Document format | Obsidian Markdown | Directly works with the target knowledge-base format |
| Chunking | Custom heading-based parser | Preserves semantic section boundaries in Markdown notes |
| Embeddings | `sentence-transformers` / `all-MiniLM-L6-v2` | Small, fast, CPU-friendly embedding model |
| Dense retrieval | NumPy cosine similarity | Simple and efficient for a small vault |
| Lexical retrieval | Self-contained BM25 implementation | Captures exact terms, acronyms, names, and keywords |
| Hybrid ranking | Reciprocal Rank Fusion (RRF) | Combines dense and lexical rankings without requiring score normalization |
| Relevance filtering | Dense + BM25 relevance gate | Prevents unsupported questions from reaching the LLM |
| Generation | Groq / `openai/gpt-oss-20b` | Fast primary generation provider |
| Fallback generation | OpenAI / `gpt-4o-mini` | Provides resilience if the primary provider fails |
| UI | Streamlit | Lightweight interactive chat interface |
| Evaluation | Deterministic Python evaluation suite | Measures retrieval, gating, generation, and citation behavior |
| Deployment | Streamlit Community Cloud | Simple public deployment suitable for this MVP |

---

## Key design decisions

### 1. Hybrid retrieval instead of dense-only search

The system uses two complementary retrieval signals.

**Dense retrieval**

Semantic embeddings help retrieve conceptually related content even when the query does not use exactly the same words as the source note.

**BM25 retrieval**

Lexical search is useful for exact terms, technical names, acronyms, and keywords that may not always receive a strong semantic similarity score.

The two ranked lists are combined using **Reciprocal Rank Fusion (RRF)**.

This gives the system both semantic and keyword-based retrieval without requiring the raw dense and BM25 scores to be directly comparable.

### 2. Heading-based chunking

Obsidian notes are naturally structured around Markdown headings.

Instead of splitting every note into arbitrary fixed-size windows, the ingestion pipeline preserves heading-level sections.

This keeps related information together and allows citations to identify the specific section used by the answer.

### 3. Lightweight NumPy vector store

The demo vault is intentionally small: approximately **16 notes and 54 chunks**.

At this scale, brute-force cosine similarity is sufficient and avoids adding an external vector database or native ANN dependency.

The local index is persisted under `vector_store/`, but that directory is ignored by Git and rebuilt automatically from the included vault when needed.

This keeps the repository portable and avoids committing generated binary artifacts.

### 4. Relevance gate and abstention

Retrieval alone does not guarantee that a question is answerable from the vault.

Before calling the LLM, the system checks whether the retrieved evidence is strong enough to consider the query relevant to the knowledge base.

Clearly unsupported questions are rejected instead of being passed to the generation model.

This is important for avoiding answers based on the model's general knowledge when the requested information is not present in the vault.

### 5. Grounded generation and citations

The generation prompt instructs the model to:

- answer using the retrieved excerpts;
- avoid guessing when the evidence is insufficient;
- cite the source note used for the answer.

Citations include the note title and, where available, the relevant section.

The evaluation suite validates citations at the note level while retaining the section-level citation in the generated answer.

For example:

```text
(Note: Hybrid Search | Section: Combining Scores)
```

is evaluated as a citation to the retrieved note:

```text
Hybrid Search
```

### 6. Provider fallback

Generation first attempts the configured Groq provider.

If the primary provider fails, the application can fall back to OpenAI when an OpenAI API key is configured.

This was chosen primarily for live-demo resilience rather than because the fallback model is inherently better.

---

## Evaluation

The project includes a deterministic evaluation suite under `evaluation/`.

The current dataset contains **12 labeled queries**:

- **8 answerable questions** with expected source notes
- **4 unanswerable questions** that should be rejected by the relevance gate

The evaluation measures retrieval quality, relevance-gate behavior, generation success, and citation grounding.

### Current results

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

Detailed per-query results are available in:

```text
evaluation/results.json
```

### Interpreting the results

The retrieval benchmark is intentionally reported without tuning the system to force a perfect score.

One answerable query currently misses its expected source within the top-4 retrieval results, resulting in the **87.5% Hit@4 and MRR scores**.

At the same time, the relevance gate correctly classified all 12 benchmark queries, rejecting the four questions whose answers were not expected to be available in the vault.

All successfully generated answers contained citations, and all evaluated citations were grounded in retrieved note titles.

This highlights an important distinction between retrieval quality and generation grounding: retrieval can occasionally miss the ideal source while the system can still correctly prevent clearly unrelated questions from being answered.

---

## Repository structure

```text
obsidian-rag-assistant/
│
├── app.py
├── README.md
├── .env.example
├── .gitignore
├── .python-version
├── pyproject.toml
├── uv.lock
│
├── src/
│   ├── __init__.py
│   ├── app.py
│   ├── generate.py
│   ├── index_store.py
│   ├── ingest.py
│   └── retrieval.py
│
├── evaluation/
│   ├── __init__.py
│   ├── dataset.json
│   ├── evaluate.py
│   └── results.json
│
├── tests/
│   ├── test_evaluation.py
│   └── test_retrieval.py
│
└── vault/
    ├── General/
    └── Personal/
```

---

## Running locally

### 1. Clone the repository

```bash
git clone https://github.com/jmhasan1/obsidian-rag-assistant.git
cd obsidian-rag-assistant
```

### 2. Create the environment

Using `uv`:

```bash
uv sync
```



### 3. Configure API keys

Copy the example environment file:

```bash
cp .env.example .env
```

Then add at least one supported generation provider:

```env
GROQ_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
```

The application uses Groq as the primary provider and OpenAI as a fallback when both are configured.

**Do not commit `.env` or API keys to Git.**

### 4. Start the application

From the repository root:

```bash
streamlit run app.py
```

The first run downloads the embedding model and builds the local index from the included `vault/`.

Subsequent runs reuse the persisted local index when it is still valid.

---

## Running the evaluation

Run the complete benchmark directly from the repository root:

```bash
uv run python evaluation/evaluate.py
```

or:

```bash
uv run python -m evaluation.evaluate
```

The evaluation writes detailed results to:

```text
evaluation/results.json
```

---

## Running tests

Run the test suite:

```bash
uv run pytest
```

Current test suite:

```text
23 passed
```

Run the linter:

```bash
uv run ruff check .
```

Both checks should pass before submitting changes.

---

## Deploying to Streamlit Community Cloud

The repository includes a root-level `app.py` launcher, so the deployed application can use:

```text
app.py
```

as its Streamlit entry point.

### Deployment steps

1. Push the repository to GitHub.
2. Create a new application in Streamlit Community Cloud.
3. Select this GitHub repository.
4. Set the main file to:

   ```text
   app.py
   ```

5. Add the required API keys under the deployment's Secrets configuration:

   ```toml
   GROQ_API_KEY = "your_key_here"
   OPENAI_API_KEY = "your_key_here"
   ```

6. Deploy the application.

The vector index does not need to be committed to Git. The application can build it from the included `vault/` during startup.

---

## Demo vault

The included demo vault contains approximately **16 Markdown notes** across `General` and `Personal` categories.

The notes cover topics including:

- RAG
- embeddings
- hybrid search
- chunking
- agentic AI
- vector databases
- MLOps
- prompt engineering
- hardware constraints
- project and career-related notes

This makes it possible to test both technical and personal/project-oriented questions against known source material.

### Example questions

```text
How does Reciprocal Rank Fusion combine retrieval results?

What embedding model is used for the local retrieval baseline?

Why is heading-based chunking a good fit for an Obsidian vault?

What are the core components of an agentic AI system?

What is the capital of France?
```

The last question should be rejected because the answer is not expected to exist in the vault.

---

## Limitations

This is an MVP designed around a small static Obsidian vault.

Current limitations include:

- No real-time synchronization with an active Obsidian vault.
- The demo corpus is intentionally small.
- Brute-force dense retrieval is appropriate for this corpus but would not be the preferred approach for very large vaults.
- Retrieval quality can still miss the ideal source within the selected top-k.
- The application relies on external LLM APIs for answer generation.
- There is no authentication or multi-user access control.
- Conversation memory is limited to the current application session.

---

## Future improvements

These are intentionally outside the current MVP submission scope and can be added in a later iteration:

- ANN/vector database backends such as FAISS or Qdrant
- MMR/source diversification
- Cross-encoder or model-based reranking
- Larger and more diverse evaluation datasets
- LLM-as-a-judge evaluation
- Metadata and folder filtering
- Multi-vault support
- Real-time Obsidian synchronization
- Obsidian plugin integration
- Graph-based note relationship exploration
- Persistent conversational memory
- Authentication and multi-user deployment
- Writing generated notes back into the vault
