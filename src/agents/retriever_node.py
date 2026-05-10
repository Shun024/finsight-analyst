"""
Retriever Agent: runs hybrid search using the router's optimised query.
"""

from src.agents.state import AgentState
from src.retrieval.hybrid_retriever import HybridRetriever

# Load once at module level — expensive to reload
_retriever = None


def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
        _retriever.load()
    return _retriever


def retriever_node(state: AgentState) -> AgentState:
    """Run hybrid retrieval using the router's optimised query."""
    print(f"\n[Retriever] Searching: {state['retrieval_query']}")

    retriever = get_retriever()
    chunks = retriever.search(
        query=state["retrieval_query"],
        top_k=6,
        filter_company=state.get("filter_company"),
        filter_year=state.get("filter_year"),
    )

    print(f"[Retriever] Retrieved {len(chunks)} chunks from: "
          f"{set(c['company'] for c in chunks)}")

    return {
        **state,
        "retrieved_chunks": chunks,
    }