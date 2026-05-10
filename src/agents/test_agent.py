"""Quick smoke test for the full agent pipeline."""

from src.agents.graph import run_query

TEST_QUERIES = [
    "What was Lloyds Banking Group's CET1 ratio in 2023?",
    "Compare Barclays and Lloyds net interest income for 2024.",
    "What are the key risks identified by the Bank of England in November 2024?",
]

if __name__ == "__main__":
    for query in TEST_QUERIES:
        print("\n" + "=" * 60)
        print(f"QUERY: {query}")
        print("=" * 60)

        result = run_query(query)

        print("\nFINAL ANSWER:")
        print(result["final_answer"])
        print(f"\nCitations: {len(result['citations'])} sources")
        print(f"Confidence: {result['confidence']}")