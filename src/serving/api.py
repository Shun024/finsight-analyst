"""
FinSight Analyst — FastAPI REST API
Production serving layer for the LangGraph agent.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import time

from src.agents.graph import run_query

# App setup
app = FastAPI(
    title="FinSight Analyst API",
    description="Agentic RAG for UK financial document intelligence",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Schemas ---

class QueryRequest(BaseModel):
    question: str
    filter_company: str | None = None
    filter_year: str | None = None


class Citation(BaseModel):
    company: str
    doc_type: str
    year: str
    page: int
    source_file: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    confidence: str
    query_type: str
    citations: list[Citation]
    chunks_retrieved: int
    latency_seconds: float
    timestamp: str


# --- Endpoints ---

@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.get("/")
def root():
    """API info."""
    return {
        "name": "FinSight Analyst API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    Run a question through the full LangGraph agent pipeline.
    Returns a structured answer with citations and confidence score.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if len(request.question) > 500:
        raise HTTPException(
            status_code=400,
            detail="Question too long — max 500 characters"
        )

    start = time.time()

    try:
        result = run_query(request.question)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Agent pipeline error: {str(e)}"
        )

    latency = round(time.time() - start, 2)

    # Clean confidence banner from answer
    answer = result.get("final_answer", "")
    if "\n\n" in answer:
        answer = answer.split("\n\n", 1)[-1]

    citations = [
        Citation(
            company=c["company"],
            doc_type=c["doc_type"],
            year=c["year"],
            page=c["page"],          # changed from page_number
            source_file=c["source_file"],
        )
        for c in result.get("citations", [])
    ]

    return QueryResponse(
        question=request.question,
        answer=answer,
        confidence=result.get("confidence", "medium"),
        query_type=result.get("query_type", "factual"),
        citations=citations,
        chunks_retrieved=len(result.get("retrieved_chunks", [])),
        latency_seconds=latency,
        timestamp=datetime.utcnow().isoformat(),
    )


@app.get("/corpus")
def corpus():
    """Return information about the document corpus."""
    return {
        "documents": [
            {"company": "Lloyds Banking Group", "doc_type": "Annual Report", "years": ["2023", "2024"]},
            {"company": "Barclays", "doc_type": "Annual Report", "years": ["2024"]},
            {"company": "NatWest", "doc_type": "Annual Report", "years": ["2023"]},
            {"company": "Bank of England", "doc_type": "Monetary Policy Report", "years": ["2024"]},
            {"company": "FCA", "doc_type": "Annual Report", "years": ["2023-24"]},
        ],
        "total_chunks": 9539,
        "embedding_model": "text-embedding-3-small",
        "retrieval": "hybrid (dense + BM25)",
    }