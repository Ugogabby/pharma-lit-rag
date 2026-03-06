"""
ingestion/pubmed_fetcher.py
───────────────────────────
Fetches paper metadata from PubMed using the NCBI Entrez E-utilities API.

What this file does:
  1. Searches PubMed with your research queries (TTE, competing events,
     opioid-gabapentinoid, OSORD, pharmacoepi methods, and more)
  2. Collects all unique PMIDs across all queries
  3. Fetches full metadata for each paper in batches of 200
  4. Saves everything to data/papers.jsonl — one paper per line
  5. Skips papers already in the manifest so re-runs are safe

This is the broadest possible net — TTE methodology papers from any drug
class, not just opioids, so the KB is useful to all pharmacoepidemiologists.
"""

import os
import json
import time
import requests
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

from config import cfg


# ── PubMed API endpoints ──────────────────────────────────────────────────────
# esearch: text query → list of PMIDs
# efetch:  list of PMIDs → full metadata XML
ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


# ── Research queries ──────────────────────────────────────────────────────────
# Organized into 8 thematic categories.
# Each string is sent to PubMed exactly as you would type it in the search box.
# [MeSH] = Medical Subject Heading — PubMed controlled vocabulary, very precise
# [tiab] = searches title AND abstract — broader, catches newer papers

QUERIES = {

    # ── 1. Target Trial Emulation — methodology (any drug class) ─────────────
    "tte_methodology": [
        "target trial emulation pharmacoepidemiology",
        "emulation target trial observational data",
        "Hernan Robins target trial",
        "trial emulation causal inference epidemiology",
        "protocol emulation observational study design",
        "target trial framework real world evidence",
        "emulated trial claims data",
        "target trial emulation cardiovascular",
        "target trial emulation diabetes",
        "target trial emulation cancer",
        "target trial emulation infectious disease",
        "target trial emulation mental health",
    ],

    # ── 2. Sequential Trial Emulation & Clone-Censor-Weight ──────────────────
    "ste_ccw": [
        "sequential target trial emulation",
        "clone censor weight method pharmacoepidemiology",
        "sequential emulation sustained treatment strategies",
        "artificial censoring IPCW pharmacoepidemiology",
        "cloning method observational sustained strategies",
        "per protocol effect observational study IPCW",
        "sequential trials pooled logistic regression",
        "treatment strategies sustained observational",
    ],

    # ── 3. Competing Events Methodology ──────────────────────────────────────
    "competing_events": [
        "competing risks pharmacoepidemiology methods",
        "Fine Gray subdistribution hazard ratio drug study",
        "cause specific hazard competing events observational",
        "competing events target trial emulation",
        "cumulative incidence competing risks pharmacoepi",
        "death competing event drug safety study",
        "subdistribution hazard cause specific hazard comparison",
        "competing risks estimand clinical trial",
        "composite outcome competing events real world",
        "while alive estimand competing events",
    ],

    # ── 4. Causal Inference & Estimands ──────────────────────────────────────
    "causal_inference": [
        "estimand framework pharmacoepidemiology",
        "ITT per protocol estimand observational study",
        "intention to treat per protocol effect real world",
        "causal inference observational pharmacoepidemiology",
        "directed acyclic graph pharmacoepidemiology",
        "confounding bias observational drug study",
        "immortal time bias pharmacoepidemiology correction",
        "time varying confounding marginal structural model",
        "inverse probability weighting treatment observational",
        "propensity score pharmacoepidemiology methods",
    ],

    # ── 5. Study Design — Active Comparator & New User ───────────────────────
    "study_design": [
        "active comparator new user design pharmacoepidemiology",
        "new user design drug safety study",
        "active comparator design confounding indication",
        "channeling bias pharmacoepidemiology active comparator",
        "prevalent user bias new user design",
        "hdPS high dimensional propensity score",
        "negative control outcome pharmacoepidemiology",
        "positive control study design drug safety",
    ],

    # ── 6. Opioid-Gabapentinoid OSORD (your specific research) ───────────────
    "opioid_gaba_osord": [
        "gabapentin opioid concurrent use overdose",
        "pregabalin opioid concurrent use overdose",
        "gabapentinoid opioid co-prescription mortality",
        "opioid sedative overdose risk CNS depressant",
        "gabapentin misuse abuse opioid",
        "pregabalin opioid respiratory depression death",
        "opioid gabapentinoid pharmacoepidemiology claims",
        "gabapentin opioid Medicaid overdose",
        "opioid benzodiazepine gabapentin concurrent",
        "CNS depressant combination overdose risk real world",
        "gabapentin opioid related death population based",
        "Gomes gabapentin opioid mortality",
    ],

    # ── 7. Administrative Claims Data Methods ────────────────────────────────
    "claims_methods": [
        "Medicaid claims pharmacoepidemiology methods",
        "Medicare claims drug safety study design",
        "T-MSIS Medicaid data opioid study",
        "administrative claims database pharmacoepidemiology",
        "ICD-10 coding pharmacoepidemiology validation",
        "days supply overlap claims data drug exposure",
        "NDC national drug code claims opioid",
        "claims based cohort opioid study",
        "Medicaid Medicare linked data pharmacoepidemiology",
        "real world evidence administrative data methods",
    ],

    # ── 8. Real World Evidence & Regulatory Methods ──────────────────────────
    "rwe_methods": [
        "real world evidence FDA drug evaluation",
        "real world evidence pharmacoepidemiology methods",
        "external control arm real world evidence",
        "pragmatic trial real world evidence",
        "HEOR health economics outcomes research methods",
        "comparative effectiveness research pharmacoepidemiology",
        "drug utilization study pharmacoepidemiology",
        "population based cohort drug safety",
    ],
}

# Flatten to a single list for iteration
ALL_QUERIES = [q for category in QUERIES.values() for q in category]


def search_pubmed(query: str, max_results: int = None) -> list[str]:
    """
    Sends one search query to PubMed and returns a list of PMIDs.

    A PMID is the unique number PubMed assigns to every paper.
    Example: 28272506 is the PMID for Gomes et al. 2017 (gabapentinoids + opioids).

    Args:
        query:       search string exactly as you would type in PubMed
        max_results: maximum PMIDs to return (defaults to cfg.MAX_RESULTS_PER_QUERY)

    Returns:
        list of PMID strings, e.g. ["28272506", "31152707", ...]
    """
    if max_results is None:
        max_results = cfg.MAX_RESULTS_PER_QUERY

    params = {
        "db":      "pubmed",
        "term":    query,
        "retmax":  max_results,
        "retmode": "json",
    }
    if cfg.NCBI_API_KEY:
        params["api_key"] = cfg.NCBI_API_KEY

    try:
        response = requests.get(ESEARCH_URL, params=params, timeout=30)
        response.raise_for_status()
        data  = response.json()
        pmids = data.get("esearchresult", {}).get("idlist", [])
        return pmids
    except Exception as e:
        print(f"  ⚠ Search failed for '{query[:50]}': {e}")
        return []


def fetch_paper_details(pmids: list[str]) -> list[dict]:
    """
    Takes a list of PMIDs and returns full metadata for each paper.

    PubMed accepts up to 200 PMIDs per request.
    We batch them automatically.

    Args:
        pmids: list of PMID strings

    Returns:
        list of paper dicts with standardized fields
    """
    if not pmids:
        return []

    papers     = []
    batch_size = cfg.FETCH_BATCH_SIZE  # 200 — PubMed's limit per request

    for i in tqdm(range(0, len(pmids), batch_size), desc="Fetching metadata"):
        batch = pmids[i : i + batch_size]

        params = {
            "db":      "pubmed",
            "id":      ",".join(batch),
            "retmode": "xml",
            "rettype": "abstract",
        }
        if cfg.NCBI_API_KEY:
            params["api_key"] = cfg.NCBI_API_KEY

        try:
            response = requests.get(EFETCH_URL, params=params, timeout=60)
            response.raise_for_status()
            batch_papers = _parse_pubmed_xml(response.text)
            papers.extend(batch_papers)
            # Respect PubMed rate limits: 10 req/sec with key, 3 without
            time.sleep(0.15 if cfg.NCBI_API_KEY else 0.4)
        except Exception as e:
            print(f"  ⚠ Fetch failed for batch {i}-{i+batch_size}: {e}")
            continue

    return papers


def _parse_pubmed_xml(xml_text: str) -> list[dict]:
    """
    Parses PubMed's XML response into a list of paper dicts.

    PubMed returns XML in PubMed Article Set format.
    Each <PubmedArticle> element is one paper.

    Args:
        xml_text: raw XML string from PubMed efetch

    Returns:
        list of paper dicts
    """
    papers = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        print(f"  ⚠ XML parse error: {e}")
        return []

    for article in root.findall(".//PubmedArticle"):
        try:
            paper = _extract_article_fields(article)
            if paper and paper.get("title"):
                papers.append(paper)
        except Exception:
            continue

    return papers


def _extract_article_fields(article) -> dict:
    """
    Extracts all relevant fields from one <PubmedArticle> XML element.

    Args:
        article: xml.etree.ElementTree Element for one paper

    Returns:
        standardized paper dict
    """

    def text(path, default=""):
        """Safely get text at an XML path."""
        node = article.find(path)
        return node.text.strip() if node is not None and node.text else default

    # ── PMID ──────────────────────────────────────────────────────────────────
    pmid = text(".//PMID")

    # ── Title ─────────────────────────────────────────────────────────────────
    # itertext() handles embedded XML tags like <i> for italics
    title_elem = article.find(".//ArticleTitle")
    title = "".join(title_elem.itertext()).strip() if title_elem is not None else ""

    # ── Abstract ──────────────────────────────────────────────────────────────
    # Structured abstracts have labeled sections: Background, Methods, Results
    # We preserve those labels for better chunk quality downstream
    abstract_parts = []
    for abs_elem in article.findall(".//AbstractText"):
        label = abs_elem.get("Label", "")
        body  = "".join(abs_elem.itertext()).strip()
        if label and body:
            abstract_parts.append(f"{label}: {body}")
        elif body:
            abstract_parts.append(body)
    abstract = " ".join(abstract_parts)

    # ── Authors ───────────────────────────────────────────────────────────────
    authors = []
    for author in article.findall(".//Author"):
        last  = text.__func__(article, ".//Author/LastName") if False else ""
        # Use direct find on the author element itself
        last_node  = author.find("LastName")
        first_node = author.find("ForeName")
        last  = last_node.text.strip()  if last_node  is not None and last_node.text  else ""
        first = first_node.text.strip() if first_node is not None and first_node.text else ""
        if last:
            authors.append(f"{last} {first}".strip())
    author_string = ", ".join(authors[:6])
    if len(authors) > 6:
        author_string += " et al."

    # ── Year ──────────────────────────────────────────────────────────────────
    year = text(".//PubDate/Year")
    if not year:
        medline = text(".//PubDate/MedlineDate")
        year    = medline[:4] if medline else "0000"

    # ── Journal ───────────────────────────────────────────────────────────────
    journal = text(".//Journal/Title") or text(".//MedlineTA")

    # ── DOI ───────────────────────────────────────────────────────────────────
    doi = ""
    for id_elem in article.findall(".//ArticleId"):
        if id_elem.get("IdType") == "doi" and id_elem.text:
            doi = id_elem.text.strip()
            break

    # ── PMC ID ────────────────────────────────────────────────────────────────
    # Non-empty = open access full text available via PubMed Central
    pmc_id = ""
    for id_elem in article.findall(".//ArticleId"):
        if id_elem.get("IdType") == "pmc" and id_elem.text:
            pmc_id = id_elem.text.strip()
            break

    # ── MeSH Terms ────────────────────────────────────────────────────────────
    # Controlled vocabulary terms assigned by NLM indexers
    # Useful for filtering and for enriching chunk metadata
    mesh_terms = []
    for mesh in article.findall(".//MeshHeading/DescriptorName"):
        if mesh.text:
            mesh_terms.append(mesh.text.strip())

    # ── Keywords ──────────────────────────────────────────────────────────────
    keywords = []
    for kw in article.findall(".//Keyword"):
        if kw.text:
            keywords.append(kw.text.strip())

    # ── Publication Type ──────────────────────────────────────────────────────
    # e.g. "Journal Article", "Review", "Meta-Analysis", "Clinical Trial"
    pub_types = []
    for pt in article.findall(".//PublicationType"):
        if pt.text:
            pub_types.append(pt.text.strip())

    # ── Citation string ───────────────────────────────────────────────────────
    citation = f"{author_string} ({year}). {title}. {journal}."
    if doi:
        citation += f" https://doi.org/{doi}"

    return {
        # Core identifiers
        "pmid":       pmid,
        "doi":        doi,
        "pmc_id":     pmc_id,         # non-empty = PMC full text available

        # Bibliographic fields
        "title":      title,
        "abstract":   abstract,
        "authors":    author_string,
        "year":       year,
        "journal":    journal,
        "pub_types":  pub_types,
        "mesh_terms": mesh_terms,
        "keywords":   keywords,
        "citation":   citation,

        # Pipeline tracking fields
        # status progresses: fetched → fulltext_retrieved → chunked → embedded
        "status":         "fetched",
        "fulltext_source": None,       # will be set by fulltext_retriever.py
        "added_by":        "pipeline",
        "added_date":      datetime.now().strftime("%Y-%m-%d"),
        "source":          "pubmed",
    }


def load_existing_pmids(manifest_path: str) -> set[str]:
    """
    Reads papers.jsonl and returns the set of PMIDs already downloaded.
    Allows safe re-runs without re-fetching existing papers.

    Args:
        manifest_path: path to papers.jsonl

    Returns:
        set of PMID strings
    """
    existing = set()
    path = Path(manifest_path)
    if not path.exists():
        return existing

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    paper = json.loads(line)
                    if paper.get("pmid"):
                        existing.add(str(paper["pmid"]))
                except json.JSONDecodeError:
                    continue
    return existing


def save_papers(papers: list[dict], manifest_path: str):
    """
    Appends new papers to papers.jsonl.
    JSONL = one JSON object per line. Efficient for large files and
    easy to read line-by-line without loading everything into memory.

    Args:
        papers:        list of paper dicts
        manifest_path: path to papers.jsonl
    """
    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "a", encoding="utf-8") as f:
        for paper in papers:
            f.write(json.dumps(paper) + "\n")


def run_fetch(
    queries:       list[str] = None,
    max_per_query: int       = None,
    date_filter:   str       = None,
) -> int:
    """
    Main entry point — runs the full PubMed fetch pipeline.

    Steps:
      1. Load existing PMIDs to avoid re-fetching
      2. Search PubMed with all queries, collect new PMIDs
      3. Fetch full metadata for new PMIDs in batches
      4. Save to papers.jsonl
      5. Print summary

    Args:
        queries:       override the default QUERIES list
        max_per_query: override cfg.MAX_RESULTS_PER_QUERY
        date_filter:   e.g. "2024/01/01"[PDAT] : "3000"[PDAT] for recent papers only

    Returns:
        number of new papers saved
    """
    if queries       is None: queries       = ALL_QUERIES
    if max_per_query is None: max_per_query = cfg.MAX_RESULTS_PER_QUERY

    manifest       = cfg.PAPERS_MANIFEST
    existing_pmids = load_existing_pmids(manifest)
    print(f"📚 Papers already in manifest: {len(existing_pmids)}")
    print(f"🔍 Running {len(queries)} PubMed queries...\n")

    # ── Step 1: Collect all new PMIDs ─────────────────────────────────────────
    all_new_pmids = set()
    for query in tqdm(queries, desc="Searching PubMed"):
        # Append date filter if provided (used by weekly update pipeline)
        full_query = f"{query} AND {date_filter}" if date_filter else query
        pmids      = search_pubmed(full_query, max_per_query)
        new        = [p for p in pmids
                      if p not in existing_pmids and p not in all_new_pmids]
        all_new_pmids.update(new)
        # Polite delay between queries
        time.sleep(0.35 if cfg.NCBI_API_KEY else 1.0)

    print(f"\n✓ Found {len(all_new_pmids)} new PMIDs across all queries")

    if not all_new_pmids:
        print("Nothing new to fetch. Manifest is up to date.")
        return 0

    # ── Step 2: Fetch full metadata ───────────────────────────────────────────
    print("\nFetching full metadata from PubMed...")
    new_papers = fetch_paper_details(list(all_new_pmids))

    # Extra safety: filter out any that somehow slipped through
    new_papers = [p for p in new_papers if p["pmid"] not in existing_pmids]

    # ── Step 3: Save ──────────────────────────────────────────────────────────
    save_papers(new_papers, manifest)

    # ── Step 4: Summary ───────────────────────────────────────────────────────
    open_access  = sum(1 for p in new_papers if p["pmc_id"])
    paywalled    = len(new_papers) - open_access
    total        = len(existing_pmids) + len(new_papers)

    print(f"\n{'='*50}")
    print(f"  New papers saved        : {len(new_papers)}")
    print(f"  Open access (PMC)       : {open_access}")
    print(f"  Paywalled (PDF needed)  : {paywalled}")
    print(f"  Total in manifest       : {total}")
    print(f"  Manifest location       : {manifest}")
    print(f"{'='*50}\n")

    return len(new_papers)


# ── Run directly ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_fetch()