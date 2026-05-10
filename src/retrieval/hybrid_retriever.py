"""
Hybrid retriever: combines dense (ChromaDB) + sparse (BM25) results
using Reciprocal Rank Fusion (RRF) — no score normalisation needed.
"""

from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_retriever import BM25Retriever


def reciprocal_rank_fusion(
    dense_results: list[dict],
    sparse_results: list[dict],
    k: int = 60,
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
) -> list[dict]:
    """
    Merge dense and sparse results using weighted RRF.
    RRF score = weight / (k + rank)
    Higher dense_weight = trust semantic similarity more.
    """
    scores: dict[str, float] = {}
    chunk_map: dict[str, dict] = {}

    for rank, chunk in enumerate(dense_results):
        cid = chunk["source_file"] + str(chunk["page_number"]) + chunk["text"][:50]
        scores[cid] = scores.get(cid, 0) + dense_weight / (k + rank + 1)
        chunk_map[cid] = chunk

    for rank, chunk in enumerate(sparse_results):
        cid = chunk["source_file"] + str(chunk["page_number"]) + chunk["text"][:50]
        scores[cid] = scores.get(cid, 0) + sparse_weight / (k + rank + 1)
        chunk_map[cid] = chunk

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for cid, rrf_score in ranked:
        chunk = chunk_map[cid].copy()
        chunk["rrf_score"] = round(rrf_score, 6)
        results.append(chunk)

    return results


class HybridRetriever:
    def __init__(
        self,
        persist_dir: str = "data/chroma",
        bm25_path: str = "data/processed/bm25_index.pkl",
    ):
        self.vector_store = VectorStore(persist_dir=persist_dir)
        self.bm25 = BM25Retriever()
        self.bm25_path = bm25_path

    def build(self, chunks: list[dict]) -> None:
        """Index all chunks into both retrievers."""
        print("\n[Dense] Indexing into ChromaDB...")
        self.vector_store.index_chunks(chunks)

        print("\n[Sparse] Building BM25 index...")
        self.bm25.build_index(chunks)
        self.bm25.save(self.bm25_path)

    def load(self) -> None:
        """Load BM25 from disk (ChromaDB auto-loads from persist_dir)."""
        self.bm25.load(self.bm25_path)

    def search(
        self,
        query: str,
        top_k: int = 6,
        filter_company: str | None = None,
        filter_year: str | None = None,
    ) -> list[dict]:
        """
        Hybrid search: dense + BM25 fused with RRF.
        Returns top_k most relevant chunks.
        """
        dense_results = self.vector_store.search(
            query,
            top_k=top_k * 2,
            filter_company=filter_company,
            filter_year=filter_year,
        )
        sparse_results = self.bm25.search(query, top_k=top_k * 2)

        fused = reciprocal_rank_fusion(dense_results, sparse_results)
        return fused[:top_k]