"""
FinSight Analyst — Streamlit Frontend
"""

import streamlit as st

from src.agents.graph import run_query

# Page config
st.set_page_config(
    page_title="FinSight Analyst",
    page_icon="📊",
    layout="wide",
)

# Header
st.title("📊 FinSight Analyst")
st.caption(
    "Agentic RAG system for UK financial document intelligence · "
    "Powered by LangGraph + Hybrid Retrieval"
)
st.divider()

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown("""
    **FinSight Analyst** answers questions over real UK financial documents including:
    - Lloyds Banking Group Annual Reports (2023, 2024)
    - Barclays Annual Report (2024)
    - NatWest Annual Report (2023)
    - Bank of England Monetary Policy Report (Nov 2024)
    - FCA Annual Report (2023–24)

    **Architecture:**
    - 🔍 Hybrid retrieval (dense + BM25)
    - 🤖 4-node LangGraph agent
    - ✅ Critic agent for hallucination detection
    - 📊 RAGAs evaluation: avg 0.720
    """)

    st.divider()
    st.header("Example Questions")
    examples = [
        "What was Lloyds CET1 ratio in 2023?",
        "What were Barclays strategic priorities for 2024?",
        "What risks did the Bank of England identify in November 2024?",
        "What was Barclays return on tangible equity in 2024?",
        "What was Lloyds statutory profit before tax in 2023?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state.query = ex

# Main query input
query = st.text_input(
    "Ask a question about UK financial documents",
    value=st.session_state.get("query", ""),
    placeholder="e.g. What was Lloyds CET1 ratio in 2023?",
)

if st.button("Analyse", type="primary", use_container_width=True):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Running agent pipeline..."):
            result = run_query(query)

        # Confidence badge
        confidence = result.get("confidence", "medium")
        badge = {"high": "🟢 High", "medium": "🟡 Medium", "low": "🔴 Low"}.get(
            confidence, "🟡 Medium"
        )

        # Agent trace
        with st.expander("🔍 Agent Trace", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Query Type", result.get("query_type", "—").title())
                st.metric("Confidence", badge)
            with col2:
                st.metric("Chunks Retrieved", len(result.get("retrieved_chunks", [])))
                companies = list(set(
                    c["company"] for c in result.get("retrieved_chunks", [])
                ))
                st.metric("Sources", ", ".join(companies))

        # Answer
        st.subheader("Answer")
        clean_answer = result.get("final_answer", "")
        # Strip confidence banner for display — shown separately
        if "\n\n" in clean_answer:
            clean_answer = clean_answer.split("\n\n", 1)[-1]
        st.markdown(clean_answer)

        # Sources
        if result.get("citations"):
            st.subheader("Sources")
            seen = set()
            for c in result["citations"]:
                key = f"{c['company']}_{c['doc_type']}_{c['year']}_{c['page']}"
                if key not in seen:
                    seen.add(key)
                    st.caption(
                        f"📄 {c['company']} · {c['doc_type']} · "
                        f"{c['year']} · Page {c['page']}"
                    )