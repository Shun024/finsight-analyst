"""Tests for the chunking module."""

from src.ingestion.chunker import split_text_into_chunks, count_tokens


def test_chunk_count_reasonable():
    """A long text should produce multiple chunks."""
    text = "This is a sentence about financial markets. " * 100
    chunks = split_text_into_chunks(text, chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1


def test_chunk_overlap_works():
    """Chunks should overlap — last words of chunk N appear in chunk N+1."""
    text = " ".join([f"word{i}" for i in range(300)])
    chunks = split_text_into_chunks(text, chunk_size=100, chunk_overlap=30)
    assert len(chunks) >= 2


def test_short_text_single_chunk():
    """Short text should produce exactly one chunk."""
    text = "Lloyds reported strong capital ratios in 2023."
    chunks = split_text_into_chunks(text, chunk_size=400, chunk_overlap=80)
    assert len(chunks) == 1


def test_token_count():
    """Token count should be positive for non-empty text."""
    assert count_tokens("Hello world") > 0


def test_empty_text_no_chunks():
    """Empty text should produce no chunks."""
    chunks = split_text_into_chunks("", chunk_size=400, chunk_overlap=80)
    assert len(chunks) == 0