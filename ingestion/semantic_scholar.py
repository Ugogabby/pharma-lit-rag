"""
ingestion/semantic_scholar.py
──────────────────────────────
Fetches papers from Semantic Scholar to complement PubMed coverage.

Why Semantic Scholar in addition to PubMed?
  - Covers 200M+ papers vs PubMed's ~35M (PubMed is biomedical only)
  - Includes citation and reference graphs — lets us find papers that
    CITE your key papers (forward citation search)
  - Better coverage of methods/statistics papers that may not be in PubMed
  - Provides open access PDF links directly in the API response

This module is DORMANT until your Semantic Scholar API key arrives.
The feature flag cfg.USE_SEMANTIC_SCHOLAR controls activation.
When your key arrives at uge2@pitt.edu:
  1. Open .env
  2. Replace SEMANTIC_SCHOLAR_API_KEY=pending with your real key
  3. Run this file — it activates automatically, zero code changes

Key capabilities used:
  /graph/v1/paper/search          — text search across all papers
  /graph/v1/paper/{id}/citations  — papers that CITE a given paper
  /graph/v1/paper/{id}/references — papers that a given paper CITES
  /graph/v1/paper/batch           — fetch details for multiple paper IDs
"""

import json
import time
import requests
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

from config import cfg


# ── Semantic Scholar API base URL ─────────────────────────────────────────────
S2_BASE = "https://api.semanticscholar.org/graph/v1"

# ── Fields to request for each paper ─────────────────────────────────────────
# Requesting only what we need keeps responses fast and small
PAPER_FIELDS = ",".join([
    "paperId",
    "externalIds",      # includes PubMed ID, DOI, ArXiv ID
    "title",
    "abstract",
    "authors",
    "year",
    "venue",            # journal or conference name
    "publicationTypes",
    "openAccessPdf",    # direct PDF URL if open access
    "citationCount",
    "fieldsOfStudy",
])

# ── Your Semantic Scholar search queries ──────────────────────────────────────
# Mirror the PubMed queries but adapted for S2's broader coverage
# S2 covers statistics, CS, and methods papers PubMed misses
S2_QUERIES = {

    # ── TTE Methodology ───────────────────────────────────────────────────────
    "tte_methodology": [
        "target trial emulation observational study",
        "emulation randomized trial observational data causal",
        "trial emulation pharmacoepidemiology Hernan",
        "protocol emulation real world evidence",
    ],

    # ── Sequential Trial Emulation ────────────────────────────────────────────
    "ste_ccw": [
        "sequential target trial emulation sustained strategies",
        "clone censor weight observational study",
        "artificial censoring inverse probability weighting",
        "per protocol effect observational IPCW",
    ],

    # ── Competing Events ──────────────────────────────────────────────────────
    "competing_events": [
        "competing risks cause specific subdistribution hazard",
        "Fine Gray model pharmacoepidemiology",
        "competing events estimand clinical trial",
        "cumulative incidence function competing risks",
    ],

    # ── Causal Inference Methods ──────────────────────────────────────────────
    "causal_inference": [
        "causal inference observational study methods",
        "propensity score matching pharmacoepidemiology",
        "marginal structural model time varying confounding",
        "directed acyclic graph epidemiology",
        "immortal time bias observational study correction",
    ],

    # ── Opioid-Gabapentinoid ──────────────────────────────────────────────────
    "opioid_gaba": [
        "gabapentin opioid overdose risk",
        "pregabalin opioid concurrent use mortality",
        "gabapentinoid CNS depressant combination",
    ],

    # ── Real World Evidence Methods ───────────────────────────────────────────
    "rwe_methods": [
        "real world evidence pharmacoepidemiology methods",
        "comparative effectiveness research observational",
        "active comparator new user design drug safety",
    ],
}

# Flatten to single list
ALL_S2_QUERIES = [q for cat in S2_QUERIES.values() for q in cat]

# ── Seed papers for citation expansion ────────────────────────────────────────
# These are landmark papers in your field.
# We fetch all papers that CITE these — forward citation search.
# This catches recent papers that build on the foundational methods.
SEED_PAPER_DOIS = [
    "10.1093/aje/kwv254",    # Hernan & Robins 2016 — Using Big Data to Emulate a Target Trial
    "10.1097/EDE.0000000000000197",  # Hernan 2016 — Does water kill?
    "10.1093/aje/kwx038",    # Dickerman et al. — Avoidable flaws in observational analyses
    "10.1136/bmj.j1996",     # Gomes et al. 2017 — gabapentinoids opioid mortality
    "10.1002/sim.8710",      # Young & Stensrud — Why ignorance isn't bliss (competing events)
    "10.1093/aje/kwab190",   # Hernán et al. — Target trial emulation: A framework
    "10.1097/EDE.0b013e3181ba42b3",  # Suissa — Immortal time bias
]


def _get_headers() -> dict:
    """
    Returns HTTP headers for Semantic Scholar API requests.
    Includes API key if available — higher rate limit with key.
    Without key: 1 request/second. With key: 10 requests/second.
    """
    headers = {"Accept": "application/json"}
    if cfg.SEMANTIC_SCHOLAR_API_KEY and cfg.SEMANTIC_SCHOLAR_API_KEY != "pending":
        headers["x-api-key"] = cfg.SEMANTIC_SCHOLAR_API_KEY
    return headers


def _rate_limit_sleep():
    """Pause between requests to respect Semantic Scholar rate limits."""
    if cfg.USE_SEMANTIC_SCHOLAR:
        time.sleep(0.15)   # 10 req/sec with key
    else:
        time.sleep(1.1)    # 1 req/sec without key


def search_semantic_scholar(query: str, limit: int = 100) -> list[dict]:
    """
    Searches Semantic Scholar for papers matching a text query.

    Args:
        query: search string
        limit: maximum results to return (max 100 per request)

    Returns:
        list of paper dicts with standardised fields
    """
    url    = f"{S2_BASE}/paper/search"
    params = {
        "query":  query,
        "limit":  min(limit, 100),   # S2 caps at 100 per request
        "fields": PAPER_FIELDS,
    }

    try:
        response = requests.get(url, headers=_get_headers(),
                                params=params, timeout=30)
        response.raise_for_status()
        data   = response.json()
        papers = data.get("data", [])
        return [_normalise_s2_paper(p) for p in papers if p.get("title")]
    except Exception as e:
        print(f"  ⚠ S2 search failed for '{query[:50]}': {e}")
        return []


def fetch_citations(doi: str, limit: int = 100) -> list[dict]:
    """
    Fetches papers that CITE the given paper (forward citation search).

    This is the most powerful S2 feature for your use case.
    Seeding with Hernan & Robins 2016 finds every paper that has
    cited the TTE framework — exactly the literature you want.

    Args:
        doi:   DOI of the seed paper
        limit: maximum citations to return

    Returns:
        list of citing paper dicts
    """
    # First resolve DOI to S2 paper ID
    paper_id = _doi_to_s2_id(doi)
    if not paper_id:
        return []

    url    = f"{S2_BASE}/paper/{paper_id}/citations"
    params = {
        "fields": PAPER_FIELDS,
        "limit":  min(limit, 1000),  # S2 allows up to 1000 for citations
    }

    papers = []
    offset = 0

    # Paginate through results (S2 returns max 1000 per request)
    while True:
        params["offset"] = offset
        try:
            response = requests.get(url, headers=_get_headers(),
                                    params=params, timeout=30)
            response.raise_for_status()
            data  = response.json()
            batch = data.get("data", [])
            if not batch:
                break

            # Citations are wrapped in {"citingPaper": {...}}
            for item in batch:
                paper = item.get("citingPaper", {})
                if paper.get("title"):
                    papers.append(_normalise_s2_paper(paper))

            # Check if there are more pages
            if len(batch) < 1000:
                break
            offset += 1000
            _rate_limit_sleep()

        except Exception as e:
            print(f"  ⚠ Citation fetch failed for DOI {doi}: {e}")
            break

    return papers


def fetch_references(doi: str) -> list[dict]:
    """
    Fetches papers that the given paper CITES (backward reference search).

    Useful for finding the foundational methods papers referenced by
    your seed papers — catches older landmark papers that predate
    the TTE framework but inform it.

    Args:
        doi: DOI of the paper whose references you want

    Returns:
        list of referenced paper dicts
    """
    paper_id = _doi_to_s2_id(doi)
    if not paper_id:
        return []

    url    = f"{S2_BASE}/paper/{paper_id}/references"
    params = {"fields": PAPER_FIELDS, "limit": 500}

    try:
        response = requests.get(url, headers=_get_headers(),
                                params=params, timeout=30)
        response.raise_for_status()
        data  = response.json()
        batch = data.get("data", [])

        # References are wrapped in {"citedPaper": {...}}
        papers = []
        for item in batch:
            paper = item.get("citedPaper", {})
            if paper.get("title"):
                papers.append(_normalise_s2_paper(paper))
        return papers

    except Exception as e:
        print(f"  ⚠ Reference fetch failed for DOI {doi}: {e}")
        return []


def _doi_to_s2_id(doi: str) -> str | None:
    """
    Resolves a DOI to a Semantic Scholar internal paper ID.

    S2 uses its own internal IDs. To use the citations/references
    endpoints we first need to convert DOI → S2 ID.

    Args:
        doi: paper DOI, e.g. "10.1136/bmj.j1996"

    Returns:
        S2 paper ID string, or None if not found
    """
    url = f"{S2_BASE}/paper/DOI:{doi}"
    try:
        response = requests.get(url, headers=_get_headers(),
                                params={"fields": "paperId"}, timeout=20)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json().get("paperId")
    except Exception:
        return None


def _normalise_s2_paper(paper: dict) -> dict:
    """
    Converts a Semantic Scholar paper dict to our standard format.

    S2 papers have a different structure from PubMed papers.
    This function maps S2 fields to the same keys used by pubmed_fetcher.py
    so downstream code (chunker, embedder) works identically for both sources.

    Args:
        paper: raw paper dict from S2 API

    Returns:
        standardised paper dict matching pubmed_fetcher.py output format
    """
    # ── Extract external IDs ──────────────────────────────────────────────────
    ext_ids = paper.get("externalIds") or {}
    pmid    = str(ext_ids.get("PubMed", ""))
    doi     = ext_ids.get("DOI", "")

    # ── Authors ───────────────────────────────────────────────────────────────
    authors_list = paper.get("authors") or []
    author_names = [a.get("name", "") for a in authors_list if a.get("name")]
    author_string = ", ".join(author_names[:6])
    if len(author_names) > 6:
        author_string += " et al."

    # ── Year ──────────────────────────────────────────────────────────────────
    year = str(paper.get("year") or "0000")

    # ── Journal / venue ───────────────────────────────────────────────────────
    journal = paper.get("venue") or ""

    # ── Open access PDF ───────────────────────────────────────────────────────
    oa_pdf  = paper.get("openAccessPdf") or {}
    pdf_url = oa_pdf.get("url", "")

    # ── Title and abstract ────────────────────────────────────────────────────
    title    = paper.get("title",    "") or ""
    abstract = paper.get("abstract", "") or ""

    # ── Citation string ───────────────────────────────────────────────────────
    citation = f"{author_string} ({year}). {title}. {journal}."
    if doi:
        citation += f" https://doi.org/{doi}"

    return {
        # Core identifiers
        "pmid":       pmid,            # may be empty if not in PubMed
        "doi":        doi,
        "pmc_id":     "",              # S2 doesn't provide PMC IDs directly
        "s2_id":      paper.get("paperId", ""),

        # Bibliographic fields
        "title":      title,
        "abstract":   abstract,
        "authors":    author_string,
        "year":       year,
        "journal":    journal,
        "citation":   citation,
        "mesh_terms": [],              # S2 doesn't have MeSH
        "keywords":   [],

        # S2-specific fields
        "open_access_pdf": pdf_url,   # direct PDF URL if available
        "citation_count":  paper.get("citationCount", 0),
        "fields_of_study": paper.get("fieldsOfStudy") or [],

        # Pipeline tracking
        "status":          "fetched",
        "fulltext_source": None,
        "added_by":        "pipeline",
        "added_date":      datetime.now().strftime("%Y-%m-%d"),
        "source":          "semantic_scholar",
    }


def load_existing_identifiers(manifest_path: str) -> tuple[set, set]:
    """
    Loads existing PMIDs and DOIs from the manifest.
    Used to avoid adding duplicate papers from S2 that are already
    in the KB via PubMed.

    Args:
        manifest_path: path to papers.jsonl

    Returns:
        tuple of (set of pmids, set of dois)
    """
    pmids = set()
    dois  = set()
    path  = Path(manifest_path)

    if not path.exists():
        return pmids, dois

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                paper = json.loads(line)
                if paper.get("pmid"):
                    pmids.add(str(paper["pmid"]))
                if paper.get("doi"):
                    dois.add(paper["doi"].lower())
            except json.JSONDecodeError:
                continue

    return pmids, dois


def save_papers(papers: list[dict], manifest_path: str):
    """
    Appends new papers to papers.jsonl, same format as pubmed_fetcher.py.

    Args:
        papers:        list of normalised paper dicts
        manifest_path: path to papers.jsonl
    """
    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "a", encoding="utf-8") as f:
        for paper in papers:
            f.write(json.dumps(paper) + "\n")


def run_semantic_scholar_fetch(
    queries:    list[str] = None,
    seed_dois:  list[str] = None,
    limit:      int       = 100,
) -> int:
    """
    Main entry point — runs the full Semantic Scholar fetch pipeline.

    Steps:
      1. Check if S2 is activated (key must be set and not "pending")
      2. Load existing identifiers to avoid duplicates
      3. Run text searches across all queries
      4. Run citation expansion from seed papers
      5. Run reference expansion from seed papers
      6. Deduplicate and save new papers

    Args:
        queries:   override default S2 queries
        seed_dois: override default seed paper DOIs for citation expansion
        limit:     max results per query

    Returns:
        number of new papers added
    """
    # ── Guard: only run when key is active ────────────────────────────────────
    if not cfg.USE_SEMANTIC_SCHOLAR:
        print("⏸  Semantic Scholar is pending API key.")
        print("   When your key arrives at uge2@pitt.edu:")
        print("   1. Open .env")
        print("   2. Replace SEMANTIC_SCHOLAR_API_KEY=pending with your key")
        print("   3. Re-run this file")
        return 0

    if queries   is None: queries   = ALL_S2_QUERIES
    if seed_dois is None: seed_dois = SEED_PAPER_DOIS

    manifest               = cfg.PAPERS_MANIFEST
    existing_pmids, existing_dois = load_existing_identifiers(manifest)
    print(f"📚 Papers already in manifest: {len(existing_pmids)}")

    all_new_papers = []
    seen_in_run    = set()   # deduplication within this run

    def is_duplicate(paper: dict) -> bool:
        """Returns True if this paper is already in the KB or seen this run."""
        pmid = str(paper.get("pmid", ""))
        doi  = paper.get("doi",  "").lower()
        s2id = paper.get("s2_id", "")

        if pmid and pmid in existing_pmids:  return True
        if doi  and doi  in existing_dois:   return True
        if s2id and s2id in seen_in_run:     return True
        return False

    def register(paper: dict):
        """Adds paper identifiers to seen sets."""
        if paper.get("s2_id"):
            seen_in_run.add(paper["s2_id"])

    # ── Step 1: Text search ───────────────────────────────────────────────────
    print(f"\n🔍 Running {len(queries)} Semantic Scholar text searches...")
    for query in tqdm(queries, desc="S2 text search"):
        results = search_semantic_scholar(query, limit)
        for paper in results:
            if not is_duplicate(paper):
                all_new_papers.append(paper)
                register(paper)
        _rate_limit_sleep()

    print(f"  Text search yielded {len(all_new_papers)} new papers")

    # ── Step 2: Citation expansion from seed papers ───────────────────────────
    print(f"\n🔗 Running citation expansion from {len(seed_dois)} seed papers...")
    for doi in tqdm(seed_dois, desc="Citation expansion"):
        citations = fetch_citations(doi, limit=500)
        for paper in citations:
            if not is_duplicate(paper):
                all_new_papers.append(paper)
                register(paper)
        _rate_limit_sleep()

    # ── Step 3: Reference expansion from seed papers ─────────────────────────
    print(f"\n📖 Running reference expansion from seed papers...")
    for doi in tqdm(seed_dois, desc="Reference expansion"):
        references = fetch_references(doi)
        for paper in references:
            if not is_duplicate(paper):
                all_new_papers.append(paper)
                register(paper)
        _rate_limit_sleep()

    # ── Step 4: Save ──────────────────────────────────────────────────────────
    if all_new_papers:
        save_papers(all_new_papers, manifest)

    open_access = sum(1 for p in all_new_papers if p.get("open_access_pdf"))
    print(f"\n{'='*50}")
    print(f"  New papers from Semantic Scholar : {len(all_new_papers)}")
    print(f"  With open access PDF             : {open_access}")
    print(f"{'='*50}\n")

    return len(all_new_papers)


# ── Run directly ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_semantic_scholar_fetch()