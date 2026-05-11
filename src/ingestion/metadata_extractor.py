"""
Saves processed chunks to disk as JSON for reproducibility.
"""

import json
from pathlib import Path

from src.ingestion.chunker import Chunk


def save_chunks(chunks: list[Chunk], output_dir: str | Path) -> Path:
    """Save chunks to a JSON file in the processed directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "chunks.json"
    data = [chunk.to_dict() for chunk in chunks]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(chunks)} chunks to {output_path}")
    return output_path


def load_chunks(processed_dir: str | Path) -> list[dict]:
    """Load chunks from the processed JSON file."""
    chunks_path = Path(processed_dir) / "chunks.json"

    if not chunks_path.exists():
        raise FileNotFoundError(f"No chunks.json found in {processed_dir}")

    with open(chunks_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} chunks from {chunks_path}")
    return data