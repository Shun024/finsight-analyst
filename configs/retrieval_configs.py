"""
Retrieval configurations to sweep over.
Each config is one MLflow run.
"""

RETRIEVAL_CONFIGS = [
    {
        "name": "baseline",
        "chunk_size": 300,
        "chunk_overlap": 60,
        "top_k": 6,
        "dense_weight": 0.7,
        "sparse_weight": 0.3,
    },
    {
        "name": "larger_chunks",
        "chunk_size": 400,
        "chunk_overlap": 80,
        "top_k": 6,
        "dense_weight": 0.7,
        "sparse_weight": 0.3,
    },
    {
        "name": "more_chunks",
        "chunk_size": 300,
        "chunk_overlap": 60,
        "top_k": 10,
        "dense_weight": 0.7,
        "sparse_weight": 0.3,
    },
    {
        "name": "dense_heavy",
        "chunk_size": 300,
        "chunk_overlap": 60,
        "top_k": 6,
        "dense_weight": 0.9,
        "sparse_weight": 0.1,
    },
    {
        "name": "sparse_heavy",
        "chunk_size": 300,
        "chunk_overlap": 60,
        "top_k": 6,
        "dense_weight": 0.5,
        "sparse_weight": 0.5,
    },
]