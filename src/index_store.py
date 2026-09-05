"""
Embedding + persistent vector store layer.

The dense index intentionally uses a small CPU-friendly sentence-transformer
model and NumPy brute-force cosine similarity. This is appropriate for a
small Obsidian vault and avoids unnecessary native vector-database
requirements.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from .ingest import load_vault

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_SCHEMA_VERSION = 2
DEFAULT_STORE_DIR = "vector_store"


class VaultIndex:
    def __init__(self, store_dir: str = DEFAULT_STORE_DIR):
        self.model = SentenceTransformer(EMBED_MODEL_NAME)
        self.store_dir = store_dir
        self.embeddings = None
        self.metadatas: list[dict] = []
        self.documents: list[str] = []
        self.ids: list[str] = []
        self.index_metadata: dict = {}

    @staticmethod
    def _vault_fingerprint(vault_dir: str) -> str:
        """Return a stable fingerprint of Markdown content and index inputs."""
        root = Path(vault_dir)
        digest = hashlib.sha256()

        files = sorted(root.rglob("*.md"))
        for path in files:
            relative = path.relative_to(root).as_posix()
            digest.update(relative.encode("utf-8"))
            digest.update(path.read_bytes())

        digest.update(EMBED_MODEL_NAME.encode("utf-8"))
        digest.update(str(INDEX_SCHEMA_VERSION).encode("utf-8"))
        return digest.hexdigest()

    def build(self, vault_dir: str):
        """Ingest the vault and rebuild the persistent dense index."""
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
        self.index_metadata = {
            "schema_version": INDEX_SCHEMA_VERSION,
            "embedding_model": EMBED_MODEL_NAME,
            "vault_fingerprint": self._vault_fingerprint(vault_dir),
            "chunk_count": len(chunks),
        }

        self._save()
        return len(chunks)

    def load_if_exists(self, vault_dir: str | None = None) -> bool:
        """Load a compatible index; rebuild is required when the vault changed."""
        emb_path = os.path.join(self.store_dir, "embeddings.npy")
        meta_path = os.path.join(self.store_dir, "meta.json")
        if not (os.path.exists(emb_path) and os.path.exists(meta_path)):
            return False

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            metadata = data.get("index_metadata", {})
            if metadata.get("schema_version") != INDEX_SCHEMA_VERSION:
                return False
            if metadata.get("embedding_model") != EMBED_MODEL_NAME:
                return False
            if vault_dir is not None:
                current_fingerprint = self._vault_fingerprint(vault_dir)
                if metadata.get("vault_fingerprint") != current_fingerprint:
                    return False

            embeddings = np.load(emb_path)
            ids = data["ids"]
            documents = data["documents"]
            metadatas = data["metadatas"]

            if not (
                len(embeddings) == len(ids) == len(documents) == len(metadatas)
            ):
                return False

            self.embeddings = embeddings
            self.ids = ids
            self.documents = documents
            self.metadatas = metadatas
            self.index_metadata = metadata
            return True
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return False

    def _save(self):
        os.makedirs(self.store_dir, exist_ok=True)
        np.save(os.path.join(self.store_dir, "embeddings.npy"), self.embeddings)
        with open(os.path.join(self.store_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "index_metadata": self.index_metadata,
                    "ids": self.ids,
                    "documents": self.documents,
                    "metadatas": self.metadatas,
                },
                f,
                indent=2,
            )

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-8
        return vectors / norms

    def query(self, question: str, top_k: int = 4):
        """Return top-k chunks ranked by cosine similarity."""
        if self.embeddings is None or len(self.embeddings) == 0:
            return []

        query_vec = self.model.encode([question])
        query_vec = self._normalize(query_vec)[0]
        scores = self.embeddings @ query_vec
        top_k = min(top_k, len(scores))
        top_idx = np.argsort(-scores)[:top_k]

        hits = []
        for i in top_idx:
            hits.append(
                {
                    "chunk_id": self.ids[i],
                    "text": self.documents[i],
                    "score": float(scores[i]),
                    **self.metadatas[i],
                }
            )
        return hits

    def count(self):
        return 0 if self.embeddings is None else len(self.embeddings)


if __name__ == "__main__":
    idx = VaultIndex()
    if not idx.load_if_exists("vault"):
        n = idx.build("vault")
    else:
        n = idx.count()
    print(f"Indexed {n} chunks. Store count: {idx.count()}")

    for q in [
        "What is hybrid search and why does it matter?",
        "What hardware do I use for local development?",
    ]:
        print(f"\nQuery: {q}")
        for h in idx.query(q, top_k=3):
            print(f"  [{h['score']:.3f}] {h['note_title']} > {h['heading']}")
