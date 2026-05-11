"""
MLflow experiment runner.
Sweeps retrieval configurations and logs RAGAs scores for each.
"""

import asyncio
import json
import os

import mlflow
from dotenv import load_dotenv
from openai import AsyncOpenAI
from ragas.embeddings import OpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

from configs.retrieval_configs import RETRIEVAL_CONFIGS
from src.agents.graph import run_query
from src.retrieval.hybrid_retriever import HybridRetriever

load_dotenv()


def load_qa_pairs(path: str = "src/evaluation/qa_pairs.json") -> list[dict]:
    with open(path) as f:
        return json.load(f)


async def score_sample(
    metrics: list,
    metric_names: list,
    q: str,
    a: str,
    ctx: list,
    ref: str,
) -> dict:
    """Score a single QA sample across all metrics."""
    clean_answer = a.split("\n\n", 1)[-1] if "\n\n" in a else a
    scores = {}

    try:
        scores["faithfulness"] = float(
            await metrics[0].ascore(
                user_input=q, response=clean_answer, retrieved_contexts=ctx
            )
        )
    except Exception:
        scores["faithfulness"] = 0.0

    try:
        scores["answer_relevancy"] = float(
            await metrics[1].ascore(user_input=q, response=clean_answer)
        )
    except Exception:
        scores["answer_relevancy"] = 0.0

    try:
        scores["context_precision"] = float(
            await metrics[2].ascore(
                user_input=q, reference=ref, retrieved_contexts=ctx
            )
        )
    except Exception:
        scores["context_precision"] = 0.0

    try:
        scores["context_recall"] = float(
            await metrics[3].ascore(
                user_input=q, retrieved_contexts=ctx, reference=ref
            )
        )
    except Exception:
        scores["context_recall"] = 0.0

    return scores


async def run_experiment(config: dict, qa_pairs: list[dict]) -> dict:
    """Run one retrieval config and return avg RAGAs scores."""
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
    metric_names = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]

    all_scores = {name: [] for name in metric_names}

    for i, qa in enumerate(qa_pairs, 1):
        print(f"  [{i}/{len(qa_pairs)}] {qa['question'][:50]}...")
        result = run_query(qa["question"])

        sample_scores = await score_sample(
            metrics=metrics,
            metric_names=metric_names,
            q=qa["question"],
            a=result["final_answer"],
            ctx=[c["text"] for c in result["retrieved_chunks"]],
            ref=qa["ground_truth"],
        )

        for name, score in sample_scores.items():
            all_scores[name].append(score)

    return {
        name: round(sum(v) / len(v), 4) if v else 0.0
        for name, v in all_scores.items()
    }


def run_mlflow_sweep():
    """Run all configs and log to MLflow."""
    mlflow.set_experiment("finsight-retrieval-sweep")
    qa_pairs = load_qa_pairs()

    print("=" * 50)
    print("FinSight MLflow Experiment Sweep")
    print(f"Configs: {len(RETRIEVAL_CONFIGS)} | QA pairs: {len(qa_pairs)}")
    print("=" * 50)

    for config in RETRIEVAL_CONFIGS:
        print(f"\nRunning config: {config['name']}")

        with mlflow.start_run(run_name=config["name"]):
            # Log hyperparameters
            mlflow.log_params({
                "chunk_size": config["chunk_size"],
                "chunk_overlap": config["chunk_overlap"],
                "top_k": config["top_k"],
                "dense_weight": config["dense_weight"],
                "sparse_weight": config["sparse_weight"],
            })

            # Run evaluation
            scores = asyncio.run(run_experiment(config, qa_pairs))

            # Log metrics
            mlflow.log_metrics(scores)
            mlflow.log_metric(
                "overall_avg",
                round(sum(scores.values()) / len(scores), 4),
            )

            print(f"  Scores: {scores}")
            print(f"  Overall: {round(sum(scores.values())/len(scores), 4)}")

    print("\nSweep complete. Run: mlflow ui")


if __name__ == "__main__":
    run_mlflow_sweep()