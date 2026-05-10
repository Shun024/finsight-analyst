"""
PDF Parser for financial documents.
Extracts text page-by-page with metadata using PyMuPDF.
"""

import fitz  # pymupdf
from pathlib import Path
from dataclasses import dataclass


@dataclass
class ParsedPage:
    text: str
    page_number: int
    source_file: str
    company: str
    doc_type: str
    year: str


def extract_company_metadata(filename: str) -> dict:
    """
    Infer company, doc_type, and year from filename.
    """
    name = Path(filename).stem.lower()
    parts = name.split("-") + name.split("_")

    # Company detection
    company_map = {
        "lloyds": "Lloyds Banking Group",
        "lbg": "Lloyds Banking Group",
        "barclays": "Barclays",
        "hsbc": "HSBC",
        "natwest": "NatWest",
        "boe": "Bank of England",
        "monetary": "Bank of England",
        "fca": "FCA",
        "annual-report-2023-24": "FCA",
        "240226": "NatWest",
    }
    company = next(
        (v for k, v in company_map.items() if k in name),
        "Unknown"
    )

    # Doc type detection
    if "annual" in name or "accounts" in name:
        doc_type = "Annual Report"
    elif "monetary" in name:
        doc_type = "Monetary Policy Report"
    elif "stability" in name:
        doc_type = "Financial Stability Report"
    else:
        doc_type = "Regulatory Document"

    # Year detection — check 4-digit sequences anywhere in filename
    import re
    years = re.findall(r"20\d{2}", name)
    year = years[0] if years else "Unknown"

    return {"company": company, "doc_type": doc_type, "year": year}


def parse_pdf(pdf_path: str | Path) -> list[ParsedPage]:
    """
    Parse a PDF file into a list of ParsedPage objects.
    Skips pages with fewer than 50 characters (covers, blank pages).
    """
    pdf_path = Path(pdf_path)
    metadata = extract_company_metadata(pdf_path.name)
    pages = []

    print(f"Parsing: {pdf_path.name}")

    with fitz.open(str(pdf_path)) as doc:
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()

            # Skip near-empty pages
            if len(text) < 50:
                continue

            pages.append(ParsedPage(
                text=text,
                page_number=page_num,
                source_file=pdf_path.name,
                company=metadata["company"],
                doc_type=metadata["doc_type"],
                year=metadata["year"],
            ))

    print(f"  → Extracted {len(pages)} pages with content")
    return pages


def parse_all_pdfs(raw_dir: str | Path) -> list[ParsedPage]:
    """Parse all PDFs in a directory."""
    raw_dir = Path(raw_dir)
    all_pages = []

    pdf_files = list(raw_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {raw_dir}")

    for pdf_file in pdf_files:
        pages = parse_pdf(pdf_file)
        all_pages.extend(pages)

    print(f"\nTotal pages extracted: {len(all_pages)}")
    return all_pages