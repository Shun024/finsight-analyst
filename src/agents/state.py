"""
Shared state object passed between all agent nodes in the graph.
LangGraph passes this through each node, each node can modify it.
"""

from typing import TypedDict, Optional


class AgentState(TypedDict):
    # Input
    query: str

    # Router output
    query_type: str          # "factual", "comparative", "analytical"
    filter_company: Optional[str]
    filter_year: Optional[str]

    # Retriever output
    retrieved_chunks: list[dict]
    retrieval_query: str     # possibly rewritten query

    # Analyst output
    answer: str
    citations: list[dict]

    # Critic output
    confidence: str          # "high", "medium", "low"
    critique: str
    final_answer: str