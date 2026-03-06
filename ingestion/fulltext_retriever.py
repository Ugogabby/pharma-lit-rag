"""
ingestion/fulltext_retriever.py
────────────────────────────────
Retrieves full text for papers in the manifest using a 4-source cascade.

For every paper, we try sources in this order and stop at the first success:

  1. PubMed Central (PMC) XML   — free, structured, best quality (~40% of papers)
  2. Unpaywall                  — finds legal free PDFs anywhere on the internet
  3. University PDF folder      — PDFs you downloaded via your institutional access
  4. Abstract only              — fallback when nothing else works

Why a cascade?
  Each source has different coverage. Running all four maximises the percentage
  of papers where we have full text rather than just an abstract. Full text
  produces 10-20x more chunks per paper, which means much richer retrieval.

Output:
  Updates each paper's record in papers.jsonl with:
    - fulltext_source: which source provided the text ("pmc", "unpaywall",
                       "local_pdf", "abstract_only")
    - fulltext_path:   path to the saved text file (in data/chunks/)
    - status:          updated to "fulltext_retrieved"
"""

import os
import json
import time
import requests
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm

from config import cfg


# ── PubMed Central full text endpoint ────────────────────────────────────────
PMC_FETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# ── Unpaywall API endpoint ────────────────────────────────────────────────────
# No key needed — just your email as identifier
# Returns metadata about where a legal free version of a paper exists
UNPAYWALL_URL = "https://api.unpaywall.org/v2/{doi}?email={email}"


# ═════════════════════════════════════════════════════════════════════════════
# SOURCE 1 — PubMed Central XML
# ═════════════════════════════════════════════════════════════════════════════

def fetch_pmc_fulltext(pmc_id: str) -> dict | None:
    """
    Fetches the full text of an open-access paper from PubMed Central.

    PMC stores papers in a structured XML format (JATS XML) that labels
    every section: Introduction, Methods, Results, Discussion, etc.
    This is the highest quality source — clean text with perfect section labels.

    Args:
        pmc_id: the PMC identifier, e.g. "PMC5395689"
                (stored in paper["pmc_id"] from the fetcher)

    Returns:
        dict with keys: sections (dict of section_name → text), source
        None if retrieval fails
    """
    if not pmc_id:
        return None

    # Strip "PMC" prefix if present — the API wants just the number
    pmc_num = pmc_id.replace("PMC", "").replace("pmc", "").strip()

    params = {
        "db":      "pmc",
        "id":      pmc_num,
        "retmode": "xml",    # JATS XML — structured full text
        "rettype": "full",
    }
    if cfg.NCBI_API_KEY:
        params["api_key"] = cfg.NCBI_API_KEY

    try:
        response = requests.get(PMC_FETCH_URL, params=params, timeout=60)
        response.raise_for_status()

        sections = _parse_pmc_xml(response.text)
        if sections and any(sections.values()):
            return {"sections": sections, "source": "pmc"}
        return None

    except Exception as e:
        return None


def _parse_pmc_xml(xml_text: str) -> dict:
    """
    Parses JATS XML from PMC and extracts text by section.

    JATS (Journal Article Tag Suite) is the XML standard used by PMC.
    Every section has a <title> tag identifying it and a <body> with the text.

    We extract these standard sections:
      abstract, introduction, methods, results, discussion, conclusion

    Args:
        xml_text: raw JATS XML string

    Returns:
        dict mapping section name → full text of that section
    """
    sections = {
        "abstract":     "",
        "introduction": "",
        "methods":      "",
        "results":      "",
        "discussion":   "",
        "conclusion":   "",
        "other":        "",
    }

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return sections

    # ── Abstract ──────────────────────────────────────────────────────────────
    abstract_parts = []
    for abs_elem in root.findall(".//abstract"):
        abstract_parts.append(_get_all_text(abs_elem))
    sections["abstract"] = " ".join(abstract_parts).strip()

    # ── Body sections ─────────────────────────────────────────────────────────
    # PMC body is divided into <sec> elements each with a <title>
    for sec in root.findall(".//body//sec"):
        title_elem = sec.find("title")
        if title_elem is None:
            continue

        # Get the section title and normalise to lowercase
        sec_title = _get_all_text(title_elem).lower().strip()
        sec_text  = _get_all_text(sec).strip()

        if not sec_text:
            continue

        # Map section title to our standard categories
        if any(kw in sec_title for kw in ["method", "material", "design",
                                           "setting", "population", "data source",
                                           "study", "statistical"]):
            sections["methods"] += " " + sec_text

        elif any(kw in sec_title for kw in ["result", "finding", "outcome"]):
            sections["results"] += " " + sec_text

        elif any(kw in sec_title for kw in ["discussion", "interpret"]):
            sections["discussion"] += " " + sec_text

        elif any(kw in sec_title for kw in ["conclusion", "summary",
                                              "implication", "recommendation"]):
            sections["conclusion"] += " " + sec_text

        elif any(kw in sec_title for kw in ["introduction", "background",
                                              "rationale", "objective"]):
            sections["introduction"] += " " + sec_text

        else:
            sections["other"] += " " + sec_text

    # Clean up extra whitespace in all sections
    return {k: " ".join(v.split()) for k, v in sections.items()}


def _get_all_text(element) -> str:
    """
    Recursively extracts all text from an XML element and its children.
    Handles nested tags like <italic>, <bold>, <sup> cleanly.

    Args:
        element: xml.etree.ElementTree Element

    Returns:
        all text content joined with spaces
    """
    parts = []
    if element.text:
        parts.append(element.text.strip())
    for child in element:
        parts.append(_get_all_text(child))
        if child.tail:
            parts.append(child.tail.strip())
    return " ".join(p for p in parts if p)


# ═════════════════════════════════════════════════════════════════════════════
# SOURCE 2 — Unpaywall
# ═════════════════════════════════════════════════════════════════════════════

def fetch_unpaywall(doi: str) -> dict | None:
    """
    Queries Unpaywall to find a legal free PDF for a paper.

    Unpaywall indexes 50M+ papers and knows where legal free versions exist:
    - Publisher open access versions
    - Institutional repository copies
    - Author posted manuscripts
    - PubMed Central (as a fallback)

    This does NOT download the PDF automatically — it returns the URL.
    The PDF download happens in fetch_and_extract_pdf() below.

    Args:
        doi: the paper's DOI, e.g. "10.1136/bmj.j1996"

    Returns:
        dict with keys: pdf_url, host_type, version, source
        None if no free version found or request fails
    """
    if not doi or not cfg.UNPAYWALL_EMAIL:
        return None

    url = UNPAYWALL_URL.format(doi=doi, email=cfg.UNPAYWALL_EMAIL)

    try:
        response = requests.get(url, timeout=20)

        # 404 means Unpaywall has no record of this DOI
        if response.status_code == 404:
            return None
        response.raise_for_status()

        data = response.json()

        # is_oa = True means a free legal version exists somewhere
        if not data.get("is_oa"):
            return None

        # best_oa_location is Unpaywall's top recommendation
        best = data.get("best_oa_location")
        if not best:
            return None

        # Prefer a direct PDF URL over a landing page URL
        pdf_url = best.get("url_for_pdf") or best.get("url")
        if not pdf_url:
            return None

        return {
            "pdf_url":   pdf_url,
            "host_type": best.get("host_type", "unknown"),  # publisher, repository, etc.
            "version":   best.get("version", "unknown"),    # publishedVersion, acceptedVersion
            "source":    "unpaywall",
        }

    except Exception:
        return None


def fetch_and_extract_pdf(pdf_url: str, pmid: str) -> dict | None:
    """
    Downloads a PDF from a URL and extracts its text using pdfplumber.

    pdfplumber extracts raw text from PDFs but does NOT label sections.
    We do a best-effort section detection based on common header patterns.

    Args:
        pdf_url: direct URL to the PDF file
        pmid:    used to name the saved PDF file

    Returns:
        dict with keys: sections, source
        None if download or extraction fails
    """
    try:
        # Download the PDF
        headers = {"User-Agent": "Mozilla/5.0 (research bot; uge2@pitt.edu)"}
        response = requests.get(pdf_url, headers=headers, timeout=60, stream=True)
        response.raise_for_status()

        # Check it is actually a PDF (not an HTML error page)
        content_type = response.headers.get("content-type", "")
        if "pdf" not in content_type.lower() and not pdf_url.endswith(".pdf"):
            # Try to proceed anyway — some servers return wrong content-type
            pass

        # Save PDF to disk
        pdf_path = Path(cfg.PDFS_DIR) / f"{pmid}_unpaywall.pdf"
        pdf_path.parent.mkdir(parents=True, exist_ok=True)

        with open(pdf_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        # Extract text using pdfplumber
        sections = _extract_pdf_sections(pdf_path)
        if sections and any(sections.values()):
            return {"sections": sections, "source": "unpaywall", "pdf_path": str(pdf_path)}
        return None

    except Exception:
        return None


def _extract_pdf_sections(pdf_path: Path) -> dict:
    """
    Extracts text from a PDF and attempts to identify sections by header patterns.

    Unlike PMC XML, PDFs have no guaranteed structure tags.
    We use heuristics: lines in ALL CAPS or matching known header patterns
    are treated as section boundaries.

    Args:
        pdf_path: path to the PDF file

    Returns:
        dict mapping section name → text
    """
    import pdfplumber

    sections = {
        "abstract":     "",
        "introduction": "",
        "methods":      "",
        "results":      "",
        "discussion":   "",
        "conclusion":   "",
        "other":        "",
    }

    # Keywords that signal each section when found in a line alone
    SECTION_MARKERS = {
        "abstract":     ["abstract"],
        "introduction": ["introduction", "background", "rationale"],
        "methods":      ["methods", "materials and methods", "study design",
                         "data sources", "patients and methods"],
        "results":      ["results", "findings"],
        "discussion":   ["discussion", "interpretation"],
        "conclusion":   ["conclusion", "conclusions", "summary",
                         "implications", "recommendations"],
    }

    current_section = "other"
    full_text_lines = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if not text:
                    continue
                full_text_lines.extend(text.split("\n"))
    except Exception:
        return sections

    for line in full_text_lines:
        line_clean = line.strip()
        line_lower = line_clean.lower()

        if not line_clean:
            continue

        # Check if this line is a section header
        matched_section = None
        for section, markers in SECTION_MARKERS.items():
            if any(line_lower == marker or line_lower.startswith(marker + " ")
                   for marker in markers):
                matched_section = section
                break

        if matched_section:
            current_section = matched_section
        else:
            sections[current_section] += " " + line_clean

    # Clean up whitespace
    return {k: " ".join(v.split()) for k, v in sections.items()}


# ═════════════════════════════════════════════════════════════════════════════
# SOURCE 3 — University PDF folder
# ═════════════════════════════════════════════════════════════════════════════

def check_local_pdf(pmid: str) -> dict | None:
    """
    Checks if you have manually downloaded a PDF for this paper.

    Workflow:
      1. You browse to the paper through your university VPN
      2. Download the PDF
      3. Rename it to {PMID}.pdf (e.g. 28272506.pdf)
      4. Drop it in data/pdfs/
      5. Next pipeline run picks it up automatically

    Args:
        pmid: the paper's PubMed ID

    Returns:
        dict with keys: sections, source, pdf_path
        None if no local PDF found
    """
    pdf_path = Path(cfg.PDFS_DIR) / f"{pmid}.pdf"

    if not pdf_path.exists():
        return None

    sections = _extract_pdf_sections(pdf_path)
    if sections and any(sections.values()):
        return {
            "sections": sections,
            "source":   "local_pdf",
            "pdf_path": str(pdf_path),
        }
    return None


# ═════════════════════════════════════════════════════════════════════════════
# SOURCE 4 — Abstract only fallback
# ═════════════════════════════════════════════════════════════════════════════

def use_abstract_only(paper: dict) -> dict:
    """
    Final fallback: use only the abstract when no full text is available.

    Even an abstract produces useful chunks for retrieval.
    The chunk will be labelled section="abstract" in the metadata.

    Args:
        paper: the paper dict from papers.jsonl

    Returns:
        dict with keys: sections, source
    """
    return {
        "sections": {
            "abstract":     paper.get("abstract", ""),
            "introduction": "",
            "methods":      "",
            "results":      "",
            "discussion":   "",
            "conclusion":   "",
            "other":        "",
        },
        "source": "abstract_only",
    }


# ═════════════════════════════════════════════════════════════════════════════
# MAIN CASCADE — tries all 4 sources in order
# ═════════════════════════════════════════════════════════════════════════════

def retrieve_fulltext(paper: dict) -> dict:
    """
    Runs the 4-source cascade for a single paper.

    Tries each source in priority order and returns the first success.
    Always returns something — worst case is abstract_only.

    Args:
        paper: paper dict from papers.jsonl
               must have: pmid, doi, pmc_id, abstract

    Returns:
        dict with keys:
          sections       — dict of section_name → text
          source         — which source provided the text
          pdf_path       — path to PDF if downloaded (optional)
    """
    pmid   = paper.get("pmid", "")
    doi    = paper.get("doi",  "")
    pmc_id = paper.get("pmc_id", "")

    # ── Source 1: PMC XML (best quality, free, structured) ───────────────────
    if pmc_id:
        result = fetch_pmc_fulltext(pmc_id)
        if result:
            return result
        time.sleep(0.2)  # brief pause before trying next source

    # ── Source 2: Unpaywall (legal free PDFs) ─────────────────────────────────
    if doi and cfg.USE_UNPAYWALL:
        unpaywall_result = fetch_unpaywall(doi)
        if unpaywall_result:
            pdf_result = fetch_and_extract_pdf(unpaywall_result["pdf_url"], pmid)
            if pdf_result:
                return pdf_result
        time.sleep(0.3)

    # ── Source 3: Local university PDF ────────────────────────────────────────
    local_result = check_local_pdf(pmid)
    if local_result:
        return local_result

    # ── Source 4: Abstract only ───────────────────────────────────────────────
    return use_abstract_only(paper)


def save_fulltext(pmid: str, fulltext: dict):
    """
    Saves extracted full text to data/chunks/{pmid}_fulltext.json

    This intermediate file stores the raw extracted text before chunking.
    The chunker (Phase 4) reads these files and splits them into chunks.

    Args:
        pmid:     the paper's PubMed ID
        fulltext: dict with sections and source
    """
    output_dir  = Path(cfg.CHUNKS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{pmid}_fulltext.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(fulltext, f, ensure_ascii=False, indent=2)


def update_manifest_status(manifest_path: str, pmid: str,
                            source: str, fulltext_path: str):
    """
    Updates a paper's record in papers.jsonl with fulltext retrieval results.

    JSONL files are append-only by design. To "update" a record we:
      1. Read all lines
      2. Modify the matching paper's fields
      3. Write everything back

    Args:
        manifest_path: path to papers.jsonl
        pmid:          the paper to update
        source:        which source provided the text
        fulltext_path: path to the saved fulltext JSON file
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
                if paper.get("pmid") == pmid:
                    paper["status"]          = "fulltext_retrieved"
                    paper["fulltext_source"] = source
                    paper["fulltext_path"]   = fulltext_path
                updated_lines.append(json.dumps(paper))
            except json.JSONDecodeError:
                updated_lines.append(line)

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(updated_lines) + "\n")


def run_fulltext_retrieval(manifest_path: str = None, limit: int = None) -> dict:
    """
    Main entry point — runs full text retrieval for all papers in the manifest
    that have not yet been retrieved.

    Args:
        manifest_path: override cfg.PAPERS_MANIFEST
        limit:         process only the first N papers (useful for testing)

    Returns:
        dict with counts: pmc, unpaywall, local_pdf, abstract_only, skipped
    """
    if manifest_path is None:
        manifest_path = cfg.PAPERS_MANIFEST

    path = Path(manifest_path)
    if not path.exists():
        print(f"Manifest not found: {manifest_path}")
        print("Run pubmed_fetcher.py first.")
        return {}

    # Load all papers that need full text retrieval
    papers_to_process = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                paper = json.loads(line)
                # Only process papers that haven't been retrieved yet
                if paper.get("status") == "fetched":
                    papers_to_process.append(paper)
            except json.JSONDecodeError:
                continue

    if limit:
        papers_to_process = papers_to_process[:limit]

    print(f"📄 Papers needing full text retrieval: {len(papers_to_process)}")
    print(f"   Unpaywall active : {cfg.USE_UNPAYWALL}")
    print(f"   Local PDF folder : {cfg.PDFS_DIR}\n")

    # ── Track source counts ───────────────────────────────────────────────────
    counts = {"pmc": 0, "unpaywall": 0, "local_pdf": 0,
              "abstract_only": 0, "skipped": 0}

    for paper in tqdm(papers_to_process, desc="Retrieving full text"):
        pmid = paper.get("pmid", "unknown")

        # Skip if fulltext file already exists
        fulltext_path = Path(cfg.CHUNKS_DIR) / f"{pmid}_fulltext.json"
        if fulltext_path.exists():
            counts["skipped"] += 1
            continue

        # Run the cascade
        result = retrieve_fulltext(paper)
        source = result.get("source", "abstract_only")
        counts[source] = counts.get(source, 0) + 1

        # Save the extracted text
        save_fulltext(pmid, result)

        # Update the manifest with retrieval status
        update_manifest_status(
            manifest_path, pmid, source, str(fulltext_path)
        )

        # Brief pause to be polite to external APIs
        time.sleep(0.1)

    # ── Summary ───────────────────────────────────────────────────────────────
    total = sum(counts.values())
    print(f"\n{'='*50}")
    print(f"  Full text retrieval complete")
    print(f"  PMC XML (best quality)  : {counts['pmc']}")
    print(f"  Unpaywall PDF           : {counts['unpaywall']}")
    print(f"  Local university PDF    : {counts['local_pdf']}")
    print(f"  Abstract only           : {counts['abstract_only']}")
    print(f"  Already done (skipped)  : {counts['skipped']}")
    print(f"  Total processed         : {total}")
    print(f"{'='*50}\n")

    return counts


# ── Run directly ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Test mode: process only first 5 papers to verify the cascade works
    import sys
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run_fulltext_retrieval(limit=limit)