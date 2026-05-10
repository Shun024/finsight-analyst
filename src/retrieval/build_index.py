"""Run this once to embed all chunks and build the retrieval index."""

from src.ingestion.metadata_extractor import load_chunks
from src.retrieval.hybrid_retriever import HybridRetriever


def main():
    print("=" * 50)
    print("FinSight Analyst — Building Retrieval Index")
    print("=" * 50)

    chunks = load_chunks("data/processed")
    retriever = HybridRetriever()
    retriever.build(chunks)

    print("\nIndex build complete. Testing a sample query...")
    retriever.load()

    results = retriever.search(
        "What is the CET1 capital ratio?",
        top_k=3,
    )

    print(f"\nTop {len(results)} results:")
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] {r['company']} — {r['doc_type']} {r['year']}")
        print(f"     RRF Score: {r['rrf_score']}")
        print(f"     Page {r['page_number']}: {r['text'][:150]}...")


if __name__ == "__main__":
    main()