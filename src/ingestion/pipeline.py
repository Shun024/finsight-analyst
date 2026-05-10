"""
Main ingestion pipeline — run this to process all PDFs.
"""

from pathlib import Path
from src.ingestion.pdf_parser import parse_all_pdfs
from src.ingestion.chunker import chunk_pages
from src.ingestion.metadata_extractor import save_chunks


def run_ingestion(
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed",
    chunk_size: int = 300,
    chunk_overlap: int = 60,
) -> None:
    print("=" * 50)
    print("FinSight Analyst — Ingestion Pipeline")
    print("=" * 50)

    # Step 1: Parse PDFs
    print("\n[1/3] Parsing PDFs...")
    pages = parse_all_pdfs(raw_dir)

    # Step 2: Chunk pages
    print("\n[2/3] Chunking pages...")
    chunks = chunk_pages(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Step 3: Save to disk
    print("\n[3/3] Saving chunks...")
    save_chunks(chunks, processed_dir)

    # Summary stats
    companies = set(c.company for c in chunks)
    print("\n" + "=" * 50)
    print("Ingestion complete.")
    print(f"  Companies: {', '.join(sorted(companies))}")
    print(f"  Total chunks: {len(chunks)}")
    print(f"  Avg chunk size: {sum(len(c.text) for c in chunks) // len(chunks)} chars")
    print("=" * 50)


if __name__ == "__main__":
    run_ingestion()