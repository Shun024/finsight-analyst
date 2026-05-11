"""
LangGraph StateGraph: wires all agent nodes into a pipeline.
"""

from langgraph.graph import END, StateGraph

from src.agents.analyst import analyst_node
from src.agents.critic import critic_node
from src.agents.retriever_node import retriever_node
from src.agents.router import router_node
from src.agents.state import AgentState


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("router", router_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("critic", critic_node)

    # Linear edges
    graph.set_entry_point("router")
    graph.add_edge("router", "retriever")
    graph.add_edge("retriever", "analyst")
    graph.add_edge("analyst", "critic")
    graph.add_edge("critic", END)

    return graph.compile()


# Singleton — compiled once and reused
agent = build_graph()


def run_query(query: str) -> dict:
    """Run a query through the full agent pipeline."""
    initial_state: AgentState = {
        "query": query,
        "query_type": "",
        "filter_company": None,
        "filter_year": None,
        "retrieved_chunks": [],
        "retrieval_query": "",
        "answer": "",
        "citations": [],
        "confidence": "",
        "critique": "",
        "final_answer": "",
    }

    result = agent.invoke(initial_state)
    return result