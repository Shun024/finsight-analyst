"""
Critic Agent: checks the analyst's answer for hallucination and confidence.
This is the key differentiator — production RAG needs self-evaluation.
"""

import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from src.agents.state import AgentState

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CRITIC_PROMPT = """You are a financial fact-checker. Review the analyst's answer against the source context.

Check for:
1. Hallucination: Does the answer contain claims NOT supported by the context?
2. Numerical accuracy: Are all figures quoted correctly from the source?
3. Attribution: Are sources correctly cited?

Return a JSON object:
{
  "confidence": "high" | "medium" | "low",
  "issues": ["list of specific issues found, or empty list if none"],
  "recommendation": "approve" | "flag"
}

confidence levels:
- high: answer is fully supported by context, no issues
- medium: mostly supported but minor gaps or uncertainties
- low: significant unsupported claims or insufficient context

Return ONLY valid JSON.
"""


def critic_node(state: AgentState) -> AgentState:
    """Evaluate the analyst's answer for hallucination and confidence."""
    print("\n[Critic] Evaluating answer quality...")

    context_summary = "\n".join([
        f"- {c['company']} {c['year']} p{c['page_number']}: "
        f"{c['text'][:100]}..."
        for c in state["retrieved_chunks"]
    ])

    user_content = f"""Query: {state['query']}

Source Context Summary:
{context_summary}

Analyst Answer:
{state['answer']}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": CRITIC_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0,
    )

    raw = response.choices[0].message.content.strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {
            "confidence": "medium",
            "issues": ["Could not parse critic response"],
            "recommendation": "flag",
        }

    confidence = parsed.get("confidence", "medium")
    issues = parsed.get("issues", [])

    print(f"[Critic] Confidence: {confidence} | "
          f"Issues: {len(issues)}")

    # Build final answer with confidence indicator
    confidence_banner = {
        "high": "✅ High Confidence",
        "medium": "⚠️ Medium Confidence",
        "low": "🔴 Low Confidence — verify independently",
    }.get(confidence, "⚠️ Medium Confidence")

    final_answer = f"{confidence_banner}\n\n{state['answer']}"

    if issues:
        final_answer += f"\n\n**Analyst Notes:** {'; '.join(issues)}"

    return {
        **state,
        "confidence": confidence,
        "critique": str(issues),
        "final_answer": final_answer,
    }