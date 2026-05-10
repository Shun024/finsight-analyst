"""
RAGAs evaluation pipeline.
Benchmarks the hybrid agent against a naive dense-only baseline.
"""

import json
import os
from datetime import datetime
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from src.agents.graph import run_query

load_dotenv()


def load_qa_pairs(path: str = "src/evaluation/qa_pairs.json") -> list[dict]:
    with open(path) as f:
        return json.load(f)


def run_benchmark(qa_pairs: list[dict]) -> dict:
    """
    Run all QA pairs through the agent and collect:
    - question
    - answer (generated)
    - contexts (retrieved chunks)
    - ground_truth
    """
    print(f"Running benchmark over {len(qa_pairs)} questions...")

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for i, qa in enumerate(qa_pairs, 1):
        print(f"\n[{i}/{len(qa_pairs)}] {qa['question'][:60]}...")

        try:
            result = run_query(qa["question"])

            questions.append(qa["question"])
            answers.append(result["final_answer"])
            contexts.append([c["text"] for c in result["retrieved_chunks"]])
            ground_truths.append(qa["ground_truth"])

        except Exception as e:
            print(f"  ERROR: {e}")
            questions.append(qa["question"])
            answers.append("Error generating answer")
            contexts.append([""])
            ground_truths.append(qa["ground_truth"])

    return {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }


def evaluate_with_ragas(data: dict) -> dict:
    """Run RAGAs metrics on the collected data."""
    import asyncio
    import os
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory
    from ragas.embeddings import OpenAIEmbeddings
    from ragas.metrics.collections import (
        Faithfulness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
    )
    from ragas.dataset_schema import SingleTurnSample

    print("\nRunning RAGAs evaluation...")

    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    llm = llm_factory("gpt-4o-mini", client=client)
    embeddings = OpenAIEmbeddings(
    client=client,
    model="text-embedding-3-small",
)

    metrics = [
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=embeddings),
        ContextPrecision(llm=llm),
        ContextRecall(llm=llm),
    ]

    async def score_all():
        results = {
            "faithfulness": [],
            "answer_relevancy": [],
            "context_precision": [],
            "context_recall": [],
        }

        metric_names = [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        ]

        for i in range(len(data["question"])):
            print(f"  Scoring sample {i+1}/{len(data['question'])}...")

            q = data["question"][i]
            a = data["answer"][i]
            ctx = data["contexts"][i]
            ref = data["ground_truth"][i]

            # Strip confidence banner added by critic (e.g. "✅ High Confidence\n\n...")
            clean_answer = a.split("\n\n", 1)[-1] if "\n\n" in a else a

            try:
                results["faithfulness"].append(
                    float(await metrics[0].ascore(
                        user_input=q, response=clean_answer, retrieved_contexts=ctx
                    ))
                )
            except Exception as e:
                print(f"    Warning: faithfulness failed — {e}")
                results["faithfulness"].append(0.0)

            try:
                results["answer_relevancy"].append(
                    float(await metrics[1].ascore(user_input=q, response=clean_answer))
                )
            except Exception as e:
                print(f"    Warning: answer_relevancy failed — {e}")
                results["answer_relevancy"].append(0.0)

            try:
                results["context_precision"].append(
                    float(await metrics[2].ascore(
                        user_input=q, reference=ref, retrieved_contexts=ctx
                    ))
                )
            except Exception as e:
                print(f"    Warning: context_precision failed — {e}")
                results["context_precision"].append(0.0)

            try:
                results["context_recall"].append(
                    float(await metrics[3].ascore(
                        user_input=q, retrieved_contexts=ctx, reference=ref
                    ))
                )
            except Exception as e:
                print(f"    Warning: context_recall failed — {e}")
                results["context_recall"].append(0.0)

        return {k: round(sum(v) / len(v), 4) if v else 0.0 for k, v in results.items()}

    return asyncio.run(score_all())


def save_results(scores: dict, output_path: str = "src/evaluation/results.json") -> None:
    """Save benchmark results with timestamp."""
    results = {
        "timestamp": datetime.now().isoformat(),
        "scores": {k: round(float(v), 4) for k, v in scores.items()},
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")


def print_results(scores: dict) -> None:
    """Pretty print the evaluation results."""
    print("\n" + "=" * 50)
    print("RAGAs Evaluation Results — FinSight Analyst")
    print("=" * 50)

    metric_descriptions = {
        "faithfulness": "Answer grounded in context (no hallucination)",
        "answer_relevancy": "Answer addresses the question",
        "context_precision": "Retrieved chunks are relevant",
        "context_recall": "All needed info was retrieved",
    }

    for metric, description in metric_descriptions.items():
        score = scores.get(metric, 0)
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"\n{metric.upper()}")
        print(f"  {description}")
        print(f"  [{bar}] {score:.3f}")

    avg = sum(float(v) for v in scores.values()) / len(scores)
    print(f"\nOVERALL AVERAGE: {avg:.3f}")
    print("=" * 50)


if __name__ == "__main__":
    qa_pairs = load_qa_pairs()
    data = run_benchmark(qa_pairs)
    scores_dict = evaluate_with_ragas(data)
    print_results(scores_dict)
    save_results(scores_dict)