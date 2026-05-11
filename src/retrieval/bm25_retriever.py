"""
BM25 sparse retriever for exact keyword matching.
Critical for financial terminology (CET1, Basel III, specific figures).
"""

import json
import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer."""
    return text.lower().split()


class BM25Retriever:
    def __init__(self):
        self.bm25 = None
        self.chunks = []

    def build_index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        print(f"Building BM25 index over {len(chunks)} chunks...")
        self.chunks = chunks
        tokenized = [tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(tokenized)
        print("BM25 index built.")

    def save(self, path: str = "data/processed/bm25_index.pkl") -> None:
        """Persist the BM25 index to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "chunks": self.chunks}, f)
        print(f"BM25 index saved to {path}")

    def load(self, path: str = "data/processed/bm25_index.pkl") -> None:
        """Load BM25 index from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.chunks = data["chunks"]
        print(f"BM25 index loaded ({len(self.chunks)} chunks)")

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """Return top_k chunks by BM25 score."""
        if self.bm25 is None:
            raise RuntimeError("BM25 index not built. Call build_index() first.")

        tokenized_query = tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Get top_k indices sorted by score
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        results = []
        for idx in top_indices:
            chunk = self.chunks[idx].copy()
            chunk["score"] = round(float(scores[idx]), 4)
            results.append(chunk)

        return results