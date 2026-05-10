"""
Chunker: splits parsed pages into overlapping chunks for embedding.
Uses token-aware splitting to respect LLM context limits.
"""

import tiktoken
from dataclasses import dataclass, asdict
from src.ingestion.pdf_parser import ParsedPage


@dataclass
class Chunk:
    chunk_id: str
    text: str
    page_number: int
    source_file: str
    company: str
    doc_type: str
    year: str
    chunk_index: int
    total_chunks_in_page: int

    def to_dict(self) -> dict:
        return asdict(self)


def count_tokens(text: str, model: str = "text-embedding-3-small") -> int:
    """Count tokens in a string."""
    # cl100k_base is used by all text-embedding-3 models
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def split_text_into_chunks(
    text: str,
    chunk_size: int = 300,
    chunk_overlap: int = 60,
) -> list[str]:
    """
    Split text into overlapping chunks by token count.
    Tries to break at sentence boundaries where possible.
    """
    enc = tiktoken.get_encoding("cl100k_base")

    # Clean up excessive whitespace from PDF extraction
    import re
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)

    tokens = enc.encode(text)

    # If text fits in one chunk, return it directly
    if len(tokens) <= chunk_size:
        return [text.strip()] if text.strip() else []

    chunks = []
    start = 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = enc.decode(chunk_tokens)

        # Try to end at a sentence boundary (. ! ?)
        # Search backwards from end for a sentence terminator
        if end < len(tokens):  # not the last chunk
            last_period = max(
                chunk_text.rfind(". "),
                chunk_text.rfind(".\n"),
                chunk_text.rfind("! "),
                chunk_text.rfind("? "),
            )
            if last_period > len(chunk_text) * 0.5:
                # Only trim if boundary is in the second half
                chunk_text = chunk_text[:last_period + 1]

        chunk_text = chunk_text.strip()
        if chunk_text:
            chunks.append(chunk_text)

        if end == len(tokens):
            break

        # Recalculate start based on actual chunk used
        actual_tokens_used = len(enc.encode(chunk_text))
        start += max(actual_tokens_used - chunk_overlap, 1)

    return chunks


def chunk_pages(
    pages: list[ParsedPage],
    chunk_size: int = 300,
    chunk_overlap: int = 60,
) -> list[Chunk]:
    """
    Convert ParsedPage objects into Chunk objects.
    """
    all_chunks = []

    for page in pages:
        text_chunks = split_text_into_chunks(
            page.text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for i, chunk_text in enumerate(text_chunks):
            chunk_id = f"{page.source_file}_p{page.page_number}_c{i}"

            all_chunks.append(Chunk(
                chunk_id=chunk_id,
                text=chunk_text,
                page_number=page.page_number,
                source_file=page.source_file,
                company=page.company,
                doc_type=page.doc_type,
                year=page.year,
                chunk_index=i,
                total_chunks_in_page=len(text_chunks),
            ))

    print(f"Total chunks created: {len(all_chunks)}")
    return all_chunks