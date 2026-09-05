"""
Embedding + vector store layer.

Uses a small CPU-friendly sentence-transformer model (fine for both a
low-VRAM local machine and a free-tier hosted deployment — see
vault/Personal/Hardware-Constraints.md for why that constraint matters)
and a plain NumPy brute-force cosine-similarity store instead of a
dedicated vector database.

Why brute-force instead of Chroma/FAISS/etc: at this vault's scale
(tens to a few hundred chunks), an ANN index buys nothing — brute-force
cosine similarity over a NumPy matrix is fast (milliseconds) and, more
importantly, has zero native/C-extension dependencies to compile. This
sidesteps a real-world problem: chroma-hnswlib publishes no prebuilt
Windows wheels, so it always tries to compile from source there,
requiring MSVC Build Tools. See vault/General/Vector-Databases.md, which
notes brute-force is a fine default for small vaults — this file follows
that reasoning directly.
"""

import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer

from ingest import load_vault

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_STORE_DIR = "vector_store"


class VaultIndex:
    def __init__(self, store_dir: str = DEFAULT_STORE_DIR):
        self.model = SentenceTransformer(EMBED_MODEL_NAME)
        self.store_dir = store_dir
        self.embeddings = None  # np.ndarray, shape (n_chunks, dim), L2-normalized
        self.metadatas = []     # list of dicts, aligned with embeddings rows
        self.documents = []     # list of chunk text, aligned with embeddings rows
        self.ids = []           # list of chunk_id, aligned with embeddings rows

    def build(self, vault_dir: str):
        """Ingest the vault and (re)populate the in-memory store from scratch."""
        chunks = load_vault(vault_dir)
        if not chunks:
            raise ValueError(f"No chunks found in vault dir: {vault_dir}")

        texts = [f"{c.heading}\n{c.text}" for c in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=False)
        embeddings = self._normalize(embeddings)

        self.embeddings = embeddings
        self.documents = [c.text for c in chunks]
        self.ids = [c.chunk_id for c in chunks]
        self.metadatas = [
            {
                "note_title": c.note_title,
                "note_path": c.note_path,
                "heading": c.heading,
                "tags": ",".join(c.tags),
                "links": ",".join(c.links),
            }
            for c in chunks
        ]

        self._save()
        return len(chunks)

    def load_if_exists(self) -> bool:
        """Try to load a previously built index from disk. Returns True if loaded."""
        emb_path = os.path.join(self.store_dir, "embeddings.npy")
        meta_path = os.path.join(self.store_dir, "meta.json")
        if not (os.path.exists(emb_path) and os.path.exists(meta_path)):
            return False

        self.embeddings = np.load(emb_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.ids = data["ids"]
        self.documents = data["documents"]
        self.metadatas = data["metadatas"]
        return True

    def _save(self):
        os.makedirs(self.store_dir, exist_ok=True)
        np.save(os.path.join(self.store_dir, "embeddings.npy"), self.embeddings)
        with open(os.path.join(self.store_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump({
                "ids": self.ids,
                "documents": self.documents,
                "metadatas": self.metadatas,
            }, f)

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-8
        return vectors / norms

    def query(self, question: str, top_k: int = 4):
        """Return the top_k most relevant chunks for a question (brute-force cosine)."""
        if self.embeddings is None or len(self.embeddings) == 0:
            return []

        query_vec = self.model.encode([question])
        query_vec = self._normalize(query_vec)[0]

        # Cosine similarity == dot product since both sides are L2-normalized
        scores = self.embeddings @ query_vec
        top_k = min(top_k, len(scores))
        top_idx = np.argsort(-scores)[:top_k]

        hits = []
        for i in top_idx:
            hits.append({
                "chunk_id": self.ids[i],
                "text": self.documents[i],
                "score": float(scores[i]),
                **self.metadatas[i],
            })
        return hits

    def count(self):
        return 0 if self.embeddings is None else len(self.embeddings)


if __name__ == "__main__":
    idx = VaultIndex()
    n = idx.build("vault")
    print(f"Indexed {n} chunks. Store count: {idx.count()}")

    for q in ["What is hybrid search and why does it matter?",
              "What hardware do I use for local development?"]:
        print(f"\nQuery: {q}")
        for h in idx.query(q, top_k=3):
            print(f"  [{h['score']:.3f}] {h['note_title']} > {h['heading']}")
