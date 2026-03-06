"""
ingestion/pdf_watcher.py
─────────────────────────
Monitors data/pdfs/ for university-downloaded PDFs and ingests them.

Your workflow for paywalled papers:
  1. Find the paper on PubMed, note its PMID
  2. Access the full text through your University of Pittsburgh VPN
  3. Download the PDF
  4. Rename it to {PMID}.pdf  (e.g.  28272506.pdf)
  5. Drop it into data/pdfs/
  6. Run this file — it finds the PDF, extracts text, updates the manifest

Why rename to PMID?
  The PMID is the key that links the PDF to its metadata in papers.jsonl.
  Without the PMID filename, we cannot match the PDF to its paper record.
  The manifest already has title, authors, abstract, citation — we just
  need to add the full text.

This file also handles PDFs contributed by collaborators — if someone
sends you a PDF of a paper not yet in your KB, run add_unknown_pdf()
and it fetches the metadata from PubMed automatically.

Batch mode:
  Drop multiple PDFs into data/pdfs/ and run once — processes all of them.
"""

import json
import time
import requests
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

from config import cfg
from ingestion.fulltext_retriever import _extract_pdf_sections, save_fulltext


# ── PubMed metadata endpoint (reused from pubmed_fetcher) ────────────────────
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def scan_pdf_folder() -> list[Path]:
    """
    Scans data/pdfs/ and returns all PDF files found.

    Returns:
        list of Path objects for every .pdf file in the folder
    """
    pdf_dir = Path(cfg.PDFS_DIR)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    pdfs = list(pdf_dir.glob("*.pdf"))
    print(f"📁 Found {len(pdfs)} PDF files in {cfg.PDFS_DIR}")
    return pdfs


def extract_pmid_from_filename(pdf_path: Path) -> str | None:
    """
    Extracts the PMID from a PDF filename.

    Expected format: {PMID}.pdf or {PMID}_anything.pdf
    Examples:
      28272506.pdf            → "28272506"
      28272506_gomes2017.pdf  → "28272506"
      28272506_unpaywall.pdf  → skip (already processed by fulltext_retriever)

    Args:
        pdf_path: Path object for the PDF file

    Returns:
        PMID string if extractable, None otherwise
    """
    stem = pdf_path.stem   # filename without extension

    # Skip PDFs downloaded by the unpaywall pipeline — already processed
    if stem.endswith("_unpaywall"):
        return None

    # Extract the numeric part before any underscore
    parts = stem.split("_")
    candidate = parts[0].strip()

    # A valid PMID is a numeric string (typically 7-8 digits)
    if candidate.isdigit():
        return candidate

    return None


def load_manifest_pmids(manifest_path: str) -> set[str]:
    """
    Returns the set of PMIDs already in papers.jsonl.

    Args:
        manifest_path: path to papers.jsonl

    Returns:
        set of PMID strings
    """
    pmids = set()
    path  = Path(manifest_path)
    if not path.exists():
        return pmids

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                paper = json.loads(line)
                if paper.get("pmid"):
                    pmids.add(str(paper["pmid"]))
            except json.JSONDecodeError:
                continue
    return pmids


def fetch_pubmed_metadata(pmid: str) -> dict | None:
    """
    Fetches metadata for a single PMID from PubMed.

    Used when a PDF is present but the paper is not yet in papers.jsonl.
    Reuses the XML parsing logic from pubmed_fetcher.py.

    Args:
        pmid: PubMed ID string

    Returns:
        standardised paper dict, or None if fetch fails
    """
    params = {
        "db":      "pubmed",
        "id":      pmid,
        "retmode": "xml",
        "rettype": "abstract",
    }
    if cfg.NCBI_API_KEY:
        params["api_key"] = cfg.NCBI_API_KEY

    try:
        response = requests.get(EFETCH_URL, params=params, timeout=30)
        response.raise_for_status()

        # Reuse the XML parser from pubmed_fetcher
        from ingestion.pubmed_fetcher import _parse_pubmed_xml
        papers = _parse_pubmed_xml(response.text)
        return papers[0] if papers else None

    except Exception as e:
        print(f"  ⚠ Could not fetch metadata for PMID {pmid}: {e}")
        return None


def add_paper_to_manifest(paper: dict, manifest_path: str):
    """
    Appends a single paper to papers.jsonl.

    Args:
        paper:         standardised paper dict
        manifest_path: path to papers.jsonl
    """
    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(paper) + "\n")


def update_paper_fulltext_status(manifest_path: str, pmid: str,
                                  source: str, fulltext_path: str):
    """
    Updates a paper's status in papers.jsonl to reflect that full text
    has been extracted from a local PDF.

    Reads the entire manifest, updates the matching record, writes back.

    Args:
        manifest_path: path to papers.jsonl
        pmid:          paper to update
        source:        "local_pdf"
        fulltext_path: path to the saved fulltext JSON
    """
    path = Path(manifest_path)
    if not path.exists():
        return

    updated_lines = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                paper = json.loads(line)
                if str(paper.get("pmid")) == str(pmid):
                    paper["status"]          = "fulltext_retrieved"
                    paper["fulltext_source"] = source
                    paper["fulltext_path"]   = fulltext_path
                updated_lines.append(json.dumps(paper))
            except json.JSONDecodeError:
                updated_lines.append(line)

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(updated_lines) + "\n")


def process_pdf(pdf_path: Path, manifest_pmids: set[str]) -> dict:
    """
    Processes a single PDF file end to end:
      1. Extracts PMID from filename
      2. Fetches metadata from PubMed if not already in manifest
      3. Extracts text sections using pdfplumber
      4. Saves fulltext JSON to data/chunks/
      5. Updates papers.jsonl

    Args:
        pdf_path:       Path to the PDF file
        manifest_pmids: set of PMIDs already in the manifest

    Returns:
        dict with keys: pmid, status, message
    """
    # ── Step 1: Get PMID from filename ────────────────────────────────────────
    pmid = extract_pmid_from_filename(pdf_path)
    if not pmid:
        return {
            "pmid":    None,
            "status":  "skipped",
            "message": f"Could not extract PMID from filename: {pdf_path.name}",
        }

    # ── Step 2: Check if fulltext already extracted ───────────────────────────
    fulltext_path = Path(cfg.CHUNKS_DIR) / f"{pmid}_fulltext.json"
    if fulltext_path.exists():
        return {
            "pmid":    pmid,
            "status":  "skipped",
            "message": f"PMID {pmid} already has fulltext extracted",
        }

    # ── Step 3: Add to manifest if not already there ──────────────────────────
    if pmid not in manifest_pmids:
        print(f"  📥 PMID {pmid} not in manifest — fetching metadata from PubMed...")
        paper = fetch_pubmed_metadata(pmid)
        if paper:
            add_paper_to_manifest(paper, cfg.PAPERS_MANIFEST)
            manifest_pmids.add(pmid)
            print(f"     Added: {paper.get('title', '')[:70]}...")
        else:
            return {
                "pmid":    pmid,
                "status":  "error",
                "message": f"Could not fetch PubMed metadata for PMID {pmid}",
            }
        time.sleep(0.35)   # be polite to PubMed

    # ── Step 4: Extract text from PDF ────────────────────────────────────────
    sections = _extract_pdf_sections(pdf_path)

    # Check we actually got something useful
    total_text = sum(len(v) for v in sections.values())
    if total_text < 100:
        return {
            "pmid":    pmid,
            "status":  "error",
            "message": f"PDF extraction yielded too little text ({total_text} chars): {pdf_path.name}",
        }

    # ── Step 5: Save fulltext JSON ────────────────────────────────────────────
    fulltext = {
        "sections": sections,
        "source":   "local_pdf",
        "pdf_path": str(pdf_path),
    }
    save_fulltext(pmid, fulltext)

    # ── Step 6: Update manifest ───────────────────────────────────────────────
    update_paper_fulltext_status(
        cfg.PAPERS_MANIFEST, pmid, "local_pdf", str(fulltext_path)
    )

    # Count words extracted per section for the summary
    section_summary = {k: len(v.split()) for k, v in sections.items() if v}

    return {
        "pmid":            pmid,
        "status":          "success",
        "message":         f"Extracted {total_text} chars from {pdf_path.name}",
        "sections_found":  section_summary,
    }


def add_unknown_pdf(pdf_path: str | Path, pmid: str):
    """
    Adds a PDF that is NOT named by PMID.

    Use this when:
      - A collaborator sends you a PDF with a non-PMID filename
      - You downloaded a PDF but forgot to rename it
      - You want to add a paper you found outside PubMed

    Steps:
      1. You provide the PMID manually
      2. This function renames the PDF to {PMID}.pdf
      3. Then processes it normally

    Args:
        pdf_path: current path to the PDF file
        pmid:     the PubMed ID for this paper

    Example:
        from ingestion.pdf_watcher import add_unknown_pdf
        add_unknown_pdf("Downloads/some_paper.pdf", "28272506")
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        print(f"❌ File not found: {pdf_path}")
        return

    # Rename to standard PMID format
    target = Path(cfg.PDFS_DIR) / f"{pmid}.pdf"
    target.parent.mkdir(parents=True, exist_ok=True)

    if pdf_path != target:
        import shutil
        shutil.copy2(pdf_path, target)
        print(f"✓ Copied to {target}")

    # Process it
    manifest_pmids = load_manifest_pmids(cfg.PAPERS_MANIFEST)
    result = process_pdf(target, manifest_pmids)
    print(f"Result: {result['status']} — {result['message']}")


def run_pdf_watcher(verbose: bool = True) -> dict:
    """
    Main entry point — scans data/pdfs/ and processes all unprocessed PDFs.

    Run this any time you drop new PDFs into data/pdfs/.
    Safe to run repeatedly — skips already-processed files.

    Args:
        verbose: print detailed results for each PDF

    Returns:
        dict with counts: success, skipped, error
    """
    pdfs = scan_pdf_folder()

    if not pdfs:
        print(f"\n💡 No PDFs found in {cfg.PDFS_DIR}/")
        print("   To add papers via university access:")
        print("   1. Find the paper on PubMed and note its PMID")
        print("   2. Download the PDF through your Pitt VPN")
        print("   3. Rename to {{PMID}}.pdf  (e.g. 28272506.pdf)")
        print(f"   4. Drop into {cfg.PDFS_DIR}/")
        print("   5. Run this file again")
        return {"success": 0, "skipped": 0, "error": 0}

    # Load manifest once — pass to each process_pdf call
    manifest_pmids = load_manifest_pmids(cfg.PAPERS_MANIFEST)
    print(f"📚 Papers in manifest: {len(manifest_pmids)}\n")

    counts  = {"success": 0, "skipped": 0, "error": 0}
    results = []

    for pdf_path in tqdm(pdfs, desc="Processing PDFs"):
        result = process_pdf(pdf_path, manifest_pmids)
        status = result.get("status", "error")
        counts[status] = counts.get(status, 0) + 1
        results.append(result)

        if verbose and status == "success":
            sections = result.get("sections_found", {})
            section_str = ", ".join(
                f"{k}:{v}w" for k, v in sections.items() if v > 0
            )
            print(f"  ✓ PMID {result['pmid']} — {section_str}")
        elif verbose and status == "error":
            print(f"  ✗ {result['message']}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"  PDFs processed successfully : {counts['success']}")
    print(f"  Skipped (already done)      : {counts['skipped']}")
    print(f"  Errors                      : {counts['error']}")
    print(f"{'='*50}")

    if counts["error"] > 0:
        print("\n⚠ Error details:")
        for r in results:
            if r.get("status") == "error":
                print(f"  • {r['message']}")

    return counts


# ── Run directly ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_pdf_watcher()