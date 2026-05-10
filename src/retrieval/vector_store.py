"""
ChromaDB vector store: persists embeddings and enables
semantic similarity search with metadata filtering.
"""

import chromadb
from chromadb.config import Settings
from src.retrieval.embedder import Embedder


class VectorStore:
    def __init__(
        self,
        persist_dir: str = "data/chroma",
        collection_name: str = "finsight",
    ):
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.embedder = Embedder()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def index_chunks(self, chunks: list[dict], batch_size: int = 100) -> None:
        """
        Embed and index all chunks into ChromaDB.
        Skips chunks already indexed (idempotent).
        """
        # Check existing IDs to avoid re-indexing
        existing = set(self.collection.get()["ids"])
        new_chunks = [c for c in chunks if c["chunk_id"] not in existing]

        if not new_chunks:
            print("All chunks already indexed. Skipping.")
            return

        print(f"Indexing {len(new_chunks)} new chunks "
              f"({len(existing)} already exist)...")

        texts = [c["text"] for c in new_chunks]
        embeddings = self.embedder.embed_batch(texts, batch_size=batch_size)

        for i in range(0, len(new_chunks), batch_size):
            batch = new_chunks[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]

            self.collection.add(
                ids=[c["chunk_id"] for c in batch],
                embeddings=batch_embeddings,
                documents=[c["text"] for c in batch],
                metadatas=[{
                    "company": c["company"],
                    "doc_type": c["doc_type"],
                    "year": c["year"],
                    "source_file": c["source_file"],
                    "page_number": c["page_number"],
                } for c in batch],
            )

        print(f"Indexing complete. Total in store: "
              f"{self.collection.count()}")

    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_company: str | None = None,
        filter_year: str | None = None,
    ) -> list[dict]:
        """
        Semantic search with optional metadata filtering.
        Returns top_k most similar chunks.
        """
        query_embedding = self.embedder.embed_text(query)

        where = {}
        if filter_company:
            where["company"] = filter_company
        if filter_year:
            where["year"] = filter_year

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where if where else None,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks.append({
                "text": doc,
                "score": round(1 - dist, 4),  # cosine similarity
                "company": meta["company"],
                "doc_type": meta["doc_type"],
                "year": meta["year"],
                "source_file": meta["source_file"],
                "page_number": meta["page_number"],
            })

        return chunks