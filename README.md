# Obsidian Vault RAG Assistant

A Retrieval-Augmented Generation assistant that answers natural-language
questions about an Obsidian vault, grounding every answer in the actual
notes and citing sources.

Built as the MVP submission for **The Skillians' Generative AI Developer
Intern Build Sprint**.

**Live demo:** _[add your deployed Streamlit URL here]_
**Repo:** _[add your GitHub URL here]_

---

## What it does

1. Ingests a folder of Obsidian-style Markdown notes (YAML frontmatter,
   `[[wikilinks]]`, heading-based structure).
2. Splits each note into heading-level chunks and embeds them.
3. Stores embeddings in-memory/on-disk as a NumPy matrix for brute-force
   cosine-similarity retrieval.
4. On a user question, retrieves the most relevant chunks and asks an LLM
   to answer **only** from that retrieved context, citing source notes.
5. Serves this as a chat UI (Streamlit) with a "Sources" panel per answer.

## Architecture

```
vault/*.md
    │
    ▼
[ingest.py]  ── parse frontmatter, split by heading, extract [[wikilinks]]
    │
    ▼
[index_store.py] ── embed chunks (MiniLM) → store as NumPy matrix (L2-normalized)
    │
    ▼  (at query time)
[index_store.py] ── embed question → retrieve top-k similar chunks
    │
    ▼
[generate.py] ── build grounded prompt → call Groq (primary) / OpenAI (fallback)
    │
    ▼
[app.py] ── Streamlit chat UI, shows answer + cited source notes
```

## Tech stack and why

| Component | Choice | Reasoning |
|---|---|---|
| Chunking | Heading-based, custom parser | Obsidian notes are already short and single-topic; splitting at `#`/`##` boundaries keeps each chunk coherent without the complexity of semantic chunking |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) | Small (22M params), CPU-friendly — runs fine on modest local hardware and free-tier hosting, no GPU dependency |
| Vector store | Plain NumPy, brute-force cosine similarity | At this scale (tens–hundreds of chunks) an ANN index adds no speed benefit; avoids a native-compiled dependency (`chroma-hnswlib` ships no Windows wheels and requires MSVC to build from source) — one less thing to break across dev machines and deployment |
| Generation | Groq (`llama-3.1-8b-instant`), OpenAI (`gpt-4o-mini`) as fallback | Groq's free tier is fast and generous; automatic fallback keeps the live demo resilient if one provider rate-limits |
| UI | Streamlit | Fastest path from pipeline to a usable, deployable chat interface |
| Deployment | Streamlit Community Cloud | Free, public URL, sufficient for this scale of vault and traffic |

## Design decisions / scope

- **Dense-only retrieval for v1.** Hybrid (BM25 + dense, combined via
  Reciprocal Rank Fusion) is a natural upgrade — already implemented in my
  other RAG project, [DocuMind] — but was deliberately deferred here to
  keep this MVP shippable within the sprint window.
- **Heading-level chunking**, not fixed-size windows — better preserves
  each section's coherence for a note-taking corpus like Obsidian.
- **Grounded-answer prompting**: the system prompt explicitly instructs the
  model to answer only from retrieved context and to say so if the context
  is insufficient, rather than fill gaps from parametric knowledge.
- **Provider fallback**: generation tries Groq first, falls back to OpenAI
  automatically — chosen for demo reliability rather than raw capability.

## What's out of scope for this MVP (roadmap)

- Live Obsidian plugin / real-time vault sync (this reads a static folder)
- Hybrid (sparse + dense) retrieval
- Graph-view / note-relationship visualization
- Multi-vault support
- Writing answers back into the vault as new notes

## Running locally

```bash
git clone <this-repo>
cd obsidian-rag-assistant
pip install -r requirements.txt

export GROQ_API_KEY=your_key_here      # and/or OPENAI_API_KEY
streamlit run src/app.py
```

The first run downloads the embedding model and builds the vector index
from `vault/`, saving it to `vector_store/` — subsequent runs reuse the
saved index.

## Deploying (Streamlit Community Cloud)

1. Push this repo to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io), create a new app
   pointing at `app.py`.
3. In the app's **Secrets**, add:
   ```toml
   GROQ_API_KEY = "your_key_here"
   OPENAI_API_KEY = "your_key_here"
   ```
4. Deploy — the public URL is your live demo link.

## Demo vault

The included `vault/` folder is a small (~16 note) demo vault mixing
general AI/ML notes (RAG, embeddings, hybrid search, agentic AI, MLOps)
with personal project notes, so retrieval quality can be sanity-checked by
hand against known content.

[DocuMind]: https://github.com/jmhasan1
