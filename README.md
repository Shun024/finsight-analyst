# 📊 FinSight Analyst

An agentic RAG system for UK financial document intelligence. Answers complex questions over real annual reports and regulatory filings using a 4-node LangGraph agent with hybrid retrieval and hallucination detection.

**Live demo:** [HuggingFace Spaces](#) *(coming soon)*

---

## Architecture

```
User Query
    │
    ▼
┌─────────────┐
│ Router Agent│  Classifies query type, extracts company/year filters
└──────┬──────┘
       │
       ▼
┌──────────────────┐
│ Retriever Agent  │  Hybrid search: dense (OpenAI) + BM25 → RRF fusion
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  Analyst Agent   │  Synthesises answer with citations from chunks
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  Critic Agent    │  Checks hallucination, assigns confidence score
└──────────────────┘
```

**Stack:** LangGraph · ChromaDB · BM25 · OpenAI · RAGAs · FastAPI · Streamlit · Docker

---

## Benchmark Results

Evaluated on 10 hand-labelled Q&A pairs from real financial filings using RAGAs:

| Metric | Score | Description |
|---|---|---|
| **Faithfulness** | 0.796 | Answers grounded in source documents |
| **Answer Relevancy** | 0.829 | Answers address the question asked |
| **Context Recall** | 0.700 | Required information successfully retrieved |
| **Context Precision** | 0.553 | Proportion of retrieved chunks that are relevant |
| **Overall Average** | **0.720** | |

---

## Document Corpus

| Source | Document | Year |
|---|---|---|
| Lloyds Banking Group | Annual Report | 2023, 2024 |
| Barclays | Annual Report | 2024 |
| NatWest | Annual Report | 2023 |
| Bank of England | Monetary Policy Report | Nov 2024 |
| FCA | Annual Report | 2023–24 |

**9,539 chunks** · avg 1,075 chars · 300-token chunks with 60-token overlap

---

## Key Design Decisions

**Why hybrid retrieval?**
Dense embeddings capture semantic similarity but miss exact financial terminology (CET1, Basel III, specific figures). BM25 captures exact matches but misses paraphrased questions. Reciprocal Rank Fusion combines both without requiring score normalisation.

**Why a Critic agent?**
Financial services demands explainability and reliability. The Critic node independently evaluates whether the Analyst's answer is supported by retrieved context, flagging hallucinations before they reach the user. This is the architecture pattern used in production RAG systems at regulated firms.

**Why chunk at 300 tokens with 60 overlap?**
Financial PDFs are dense. Initial benchmarking at 400 tokens produced avg chunk sizes of 1,475 chars with 15.7% exceeding 2,000 chars. Reducing to 300 tokens brought 96% of chunks under 1,500 chars, improving retrieval precision.

---

## Quickstart

### Prerequisites
- Python 3.11+
- OpenAI API key (~$0.05 to index all documents)

### Setup

```bash
git clone https://github.com/Shun024/finsight-analyst.git
cd finsight-analyst

python3.11 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
# Add your OPENAI_API_KEY to .env
```

### Add Documents

Download annual reports (links in `/docs/data_sources.md`) and place PDFs in `data/raw/`.

### Run Pipeline

```bash
# 1. Parse and chunk documents
python -m src.ingestion.pipeline

# 2. Build retrieval index (~5 min, ~$0.04)
python -m src.retrieval.build_index

# 3. Launch frontend
PYTHONPATH=. streamlit run src/serving/frontend.py
```

### Docker

```bash
docker-compose up --build
```

---

## Evaluation

```bash
python -m src.evaluation.benchmark
```

Results saved to `src/evaluation/results.json`.

---

## Project Structure

```
finsight-analyst/
├── src/
│   ├── ingestion/        # PDF parsing, chunking, metadata extraction
│   ├── retrieval/        # Embedder, ChromaDB, BM25, hybrid RRF
│   ├── agents/           # LangGraph: router, retriever, analyst, critic
│   ├── evaluation/       # RAGAs benchmark, Q&A pairs
│   └── serving/          # Streamlit frontend
├── data/
│   ├── raw/              # Source PDFs (gitignored)
│   └── processed/        # Chunks JSON, BM25 index
├── tests/                # Unit tests
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Known Limitations & Future Work

- **Context Precision (0.553):** Hybrid retrieval pulls some irrelevant chunks for broad queries. Adding a cross-encoder re-ranker (e.g. `flashrank`) would improve precision.
- **Comparative queries:** Multi-company questions sometimes retrieve from only one company. A query decomposition step would split "Compare Barclays and Lloyds" into two parallel retrievals.
- **Corpus size:** 6 documents is a proof-of-concept. Production would index full FTSE 100 filings with DVC-tracked versioning.
- **Deployment:** HuggingFace Spaces deployment in progress.

---

## Author

**Shun Le Yi Mon (Sheryl)**
Data Scientist · NLP & GenAI
[LinkedIn](#) · [GitHub](https://github.com/Shun024)