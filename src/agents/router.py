"""
Router Agent: classifies the query and extracts metadata filters.
Determines how the retriever should search.
"""

import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from src.agents.state import AgentState

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

ROUTER_PROMPT = """You are a financial query router. Analyse the user query and return a JSON object with:

- query_type: one of "factual" (specific number/date/fact), "comparative" (comparing companies or years), "analytical" (trends, risks, strategy)
- filter_company: the company name if query is about ONE specific company, else null. Must be one of: "Lloyds Banking Group", "Barclays", "HSBC", "NatWest", "Bank of England", "FCA"
- filter_year: the year if query is about a specific year (e.g. "2023"), else null
- retrieval_query: an optimised search query for retrieving relevant chunks (expand abbreviations, add context)

Return ONLY valid JSON, no explanation.

Examples:
Query: "What was Lloyds CET1 ratio in 2023?"
{"query_type": "factual", "filter_company": "Lloyds Banking Group", "filter_year": "2023", "retrieval_query": "Lloyds Banking Group Common Equity Tier 1 CET1 capital ratio 2023"}

Query: "Compare Barclays and NatWest net interest margin"
{"query_type": "comparative", "filter_company": null, "filter_year": null, "retrieval_query": "net interest margin NIM comparison Barclays NatWest"}
"""


def router_node(state: AgentState) -> AgentState:
    """Classify query and extract retrieval parameters."""
    print(f"\n[Router] Analysing query: {state['query']}")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": ROUTER_PROMPT},
            {"role": "user", "content": state["query"]},
        ],
        temperature=0,
    )

    raw = response.choices[0].message.content.strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback if JSON is malformed
        parsed = {
            "query_type": "factual",
            "filter_company": None,
            "filter_year": None,
            "retrieval_query": state["query"],
        }

    print(f"[Router] Type: {parsed['query_type']} | "
          f"Company: {parsed.get('filter_company')} | "
          f"Year: {parsed.get('filter_year')}")

    return {
        **state,
        "query_type": parsed["query_type"],
        "filter_company": parsed.get("filter_company"),
        "filter_year": parsed.get("filter_year"),
        "retrieval_query": parsed["retrieval_query"],
    }