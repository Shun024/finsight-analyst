"""
Embedder: generates OpenAI embeddings for chunks.
Batches requests to stay within API rate limits.
"""

import os
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class Embedder:
    def __init__(self, model: str = "text-embedding-3-small"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.dimension = 1536

    def embed_text(self, text: str) -> list[float]:
        """Embed a single string."""
        response = self.client.embeddings.create(
            input=text,
            model=self.model,
        )
        return response.data[0].embedding

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 100,
        sleep_between: float = 0.5,
    ) -> list[list[float]]:
        """
        Embed a list of texts in batches.
        Sleeps between batches to respect rate limits.
        """
        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_num = (i // batch_size) + 1

            print(f"  Embedding batch {batch_num}/{total_batches} "
                  f"({len(batch)} texts)...")

            response = self.client.embeddings.create(
                input=batch,
                model=self.model,
            )
            batch_embeddings = [r.embedding for r in response.data]
            all_embeddings.extend(batch_embeddings)

            if i + batch_size < len(texts):
                time.sleep(sleep_between)

        return all_embeddings