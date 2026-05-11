"""
Evidently AI monitoring dashboard.
Tracks retrieval quality and answer confidence drift over time.
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset


MONITORING_LOG = Path("data/monitoring_log.json")

NUMERIC_COLS = [
    "confidence_score",
    "chunks_retrieved",
    "companies_retrieved",
    "answer_length",
    "num_citations",
    "critic_issues",
]


def log_query_result(result: dict, question: str) -> None:
    """Append a query result to the monitoring log."""
    MONITORING_LOG.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "question": question,
        "confidence": result.get("confidence", "medium"),
        "confidence_score": {"high": 1.0, "medium": 0.5, "low": 0.0}.get(
            result.get("confidence", "medium"), 0.5
        ),
        "chunks_retrieved": len(result.get("retrieved_chunks", [])),
        "query_type": result.get("query_type", "factual"),
        "companies_retrieved": len(set(
            c["company"] for c in result.get("retrieved_chunks", [])
        )),
        "answer_length": len(result.get("final_answer", "")),
        "num_citations": len(result.get("citations", [])),
        "critic_issues": 0 if result.get("critique", "[]") in ["[]", ""] else 1,
    }

    log = []
    if MONITORING_LOG.exists():
        with open(MONITORING_LOG) as f:
            log = json.load(f)

    log.append(entry)

    with open(MONITORING_LOG, "w") as f:
        json.dump(log, f, indent=2)


def load_monitoring_log() -> pd.DataFrame:
    """Load the monitoring log as a DataFrame."""
    if not MONITORING_LOG.exists():
        raise FileNotFoundError(
            f"No monitoring log found at {MONITORING_LOG}. "
            "Run some queries first."
        )

    with open(MONITORING_LOG) as f:
        log = json.load(f)

    return pd.DataFrame(log)


def generate_monitoring_report(
    output_path: str = "data/monitoring_report.html",
) -> None:
    """Generate an Evidently drift report."""
    df = load_monitoring_log()

    if len(df) < 10:
        print(f"Only {len(df)} queries logged — need at least 10.")
        return

    mid = len(df) // 2
    reference_df = df.iloc[:mid][NUMERIC_COLS].reset_index(drop=True)
    current_df = df.iloc[mid:][NUMERIC_COLS].reset_index(drop=True)

    definition = DataDefinition(numerical_columns=NUMERIC_COLS)

    reference_dataset = Dataset.from_pandas(
        reference_df,
        data_definition=definition,
    )
    current_dataset = Dataset.from_pandas(
        current_df,
        data_definition=definition,
    )

    report = Report(metrics=[DataDriftPreset()])
    my_run = report.run(
        reference_data=reference_dataset,
        current_data=current_dataset,
    )
    my_run.save_html(output_path)

    print(f"Monitoring report saved to {output_path}")
    print(f"Reference: {len(reference_df)} queries | Current: {len(current_df)} queries")


if __name__ == "__main__":
    generate_monitoring_report()