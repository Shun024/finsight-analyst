"""
Analyst Agent: synthesises retrieved chunks into a structured answer with citations.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
from src.agents.state import AgentState

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

ANALYST_PROMPT = """You are a senior financial analyst assistant. Using ONLY the provided context chunks, answer the user's question.

Rules:
- Base your answer strictly on the provided context. Do not use outside knowledge.
- Always cite your sources using [Company, DocType, Year, Page X] format.
- For numerical data, quote figures exactly as they appear in the source.
- If the context is insufficient, say so clearly.
- Structure your answer with: Summary (2-3 sentences), Key Findings (bullet points), and Sources.

Context:
{context}

Query Type: {query_type}
"""


def format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a numbered context block."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[{i}] {chunk['company']} | {chunk['doc_type']} | "
            f"{chunk['year']} | Page {chunk['page_number']}\n"
            f"{chunk['text']}"
        )
    return "\n\n---\n\n".join(parts)


def analyst_node(state: AgentState) -> AgentState:
    """Generate a structured answer from retrieved chunks."""
    print(f"\n[Analyst] Synthesising answer from "
          f"{len(state['retrieved_chunks'])} chunks...")

    context = format_context(state["retrieved_chunks"])

    system_prompt = ANALYST_PROMPT.format(
        context=context,
        query_type=state["query_type"],
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": state["query"]},
        ],
        temperature=0.1,
    )

    answer = response.choices[0].message.content.strip()

    # Extract citations from chunks
    citations = [
        {
            "company": c["company"],
            "doc_type": c["doc_type"],
            "year": c["year"],
            "page": c["page_number"],
            "source_file": c["source_file"],
        }
        for c in state["retrieved_chunks"]
    ]

    print(f"[Analyst] Answer generated ({len(answer)} chars)")

    return {
        **state,
        "answer": answer,
        "citations": citations,
    }