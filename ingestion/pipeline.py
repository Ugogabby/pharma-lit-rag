"""
ingestion/pipeline.py
──────────────────────
Master pipeline — runs all ingestion phases in sequence.

This is the single file you run to build or update your knowledge base.
It orchestrates all four ingestion modules in the correct order:

  Phase 1 — PubMed fetch
    Searches PubMed with all queries, downloads metadata for new papers,
    saves to data/papers.jsonl

  Phase 2 — Semantic Scholar fetch (when key is active)
    Searches S2 for additional papers, runs citation expansion from
    seed papers, adds new papers to data/papers.jsonl

  Phase 3 — PDF watcher
    Scans data/pdfs/ for university-downloaded PDFs, extracts text,
    updates manifest

  Phase 4 — Full text retrieval
    For every paper in the manifest, tries PMC XML → Unpaywall →
    local PDF → abstract only, saves extracted text to data/chunks/

Run modes:
  Full build   : python ingestion/pipeline.py
  Update only  : python ingestion/pipeline.py --update
  PDFs only    : python ingestion/pipeline.py --pdfs
  Fetch only   : python ingestion/pipeline.py --fetch
  Limit papers : python ingestion/pipeline.py --limit 10

The --update mode is what the weekly GitHub Actions workflow uses.
It passes a date filter to PubMed so only papers from the last 7 days
are fetched, keeping the weekly run fast.
"""

import sys
import time
import argparse
from datetime import datetime, timedelta
from pathlib import Path

from config import cfg


def print_banner(title: str):
    """Prints a formatted section banner to make pipeline output readable."""
    width = 60
    print(f"\n{'═' * width}")
    print(f"  {title}")
    print(f"{'═' * width}\n")


def get_manifest_stats() -> dict:
    """
    Returns current counts from papers.jsonl for the progress report.

    Returns:
        dict with keys: total, fetched, fulltext_retrieved, chunked, embedded
    """
    import json

    manifest = Path(cfg.PAPERS_MANIFEST)
    stats    = {
        "total":              0,
        "fetched":            0,
        "fulltext_retrieved": 0,
        "chunked":            0,
        "embedded":           0,
    }

    if not manifest.exists():
        return stats

    with open(manifest, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                import json as _json
                paper  = _json.loads(line)
                status = paper.get("status", "fetched")
                stats["total"] += 1
                if status in stats:
                    stats[status] += 1
                else:
                    stats["fetched"] += 1
            except Exception:
                continue

    return stats


def run_phase_1_pubmed(date_filter: str = None, limit: int = None):
    """
    Phase 1 — PubMed metadata fetch.

    Args:
        date_filter: e.g. "2025/01/01"[PDAT] : "3000"[PDAT] for recent only
        limit:       max results per query (for testing)
    """
    print_banner("Phase 1 — PubMed Fetch")

    from ingestion.pubmed_fetcher import run_fetch, ALL_QUERIES

    queries = ALL_QUERIES
    max_per = limit if limit else cfg.MAX_RESULTS_PER_QUERY

    new_count = run_fetch(
        queries       = queries,
        max_per_query = max_per,
        date_filter   = date_filter,
    )

    print(f"\n✓ Phase 1 complete — {new_count} new papers added from PubMed")
    return new_count


def run_phase_2_semantic_scholar(limit: int = None):
    """
    Phase 2 — Semantic Scholar fetch (only when key is active).

    Args:
        limit: max results per query (for testing)
    """
    print_banner("Phase 2 — Semantic Scholar Fetch")

    if not cfg.USE_SEMANTIC_SCHOLAR:
        print("⏸  Semantic Scholar pending API key — skipping.")
        print("   Add key to .env when it arrives at uge2@pitt.edu")
        return 0

    from ingestion.semantic_scholar import run_semantic_scholar_fetch

    new_count = run_semantic_scholar_fetch(limit=limit or 100)
    print(f"\n✓ Phase 2 complete — {new_count} new papers added from Semantic Scholar")
    return new_count


def run_phase_3_pdf_watcher():
    """
    Phase 3 — Process university-downloaded PDFs from data/pdfs/
    """
    print_banner("Phase 3 — PDF Watcher")

    from ingestion.pdf_watcher import run_pdf_watcher

    counts = run_pdf_watcher(verbose=True)
    total  = counts.get("success", 0)
    print(f"\n✓ Phase 3 complete — {total} PDFs processed")
    return total


def run_phase_4_fulltext(limit: int = None):
    """
    Phase 4 — Full text retrieval for all papers not yet retrieved.

    Args:
        limit: process only first N papers (for testing)
    """
    print_banner("Phase 4 — Full Text Retrieval")

    from ingestion.fulltext_retriever import run_fulltext_retrieval

    counts = run_fulltext_retrieval(limit=limit)

    pmc      = counts.get("pmc",           0)
    unpay    = counts.get("unpaywall",      0)
    local    = counts.get("local_pdf",      0)
    abstract = counts.get("abstract_only",  0)
    total    = pmc + unpay + local + abstract

    print(f"\n✓ Phase 4 complete — {total} papers retrieved")
    print(f"  Sources: PMC={pmc}  Unpaywall={unpay}  LocalPDF={local}  AbstractOnly={abstract}")
    return counts


def print_final_report(start_time: float, phase_results: dict):
    """
    Prints a formatted summary of the entire pipeline run.

    Args:
        start_time:    time.time() at pipeline start
        phase_results: dict of phase → result counts
    """
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    stats = get_manifest_stats()

    print_banner("Pipeline Complete")
    print(f"  Runtime                  : {minutes}m {seconds}s")
    print(f"  Completed at             : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print()
    print(f"  ── Knowledge Base Status ──────────────────")
    print(f"  Total papers in manifest : {stats['total']}")
    print(f"  Full text retrieved      : {stats['fulltext_retrieved']}")
    print(f"  Metadata only (fetched)  : {stats['fetched']}")
    print()
    print(f"  ── This Run ───────────────────────────────")
    for phase, result in phase_results.items():
        print(f"  {phase:<30} : {result}")
    print()
    print(f"  ── Next Steps ─────────────────────────────")
    print(f"  Run chunking    : python embeddings/chunker.py")
    print(f"  Run embeddings  : python embeddings/embedder.py")
    print(f"  Run app         : streamlit run app/app.py")
    print(f"{'═' * 60}\n")


def run_pipeline(
    mode:        str  = "full",
    limit:       int  = None,
    date_filter: str  = None,
    skip_s2:     bool = False,
):
    """
    Master pipeline runner.

    Args:
        mode:        "full"   — run all phases (initial build)
                     "update" — fetch only recent papers (weekly run)
                     "fetch"  — phases 1+2 only (fetch metadata, no fulltext)
                     "pdfs"   — phase 3 only (process local PDFs)
        limit:       max results per query (None = use config default)
        date_filter: PubMed date filter string for --update mode
        skip_s2:     skip Semantic Scholar even if key is active
    """
    start_time    = time.time()
    phase_results = {}

    print_banner(f"pharma-lit-rag Ingestion Pipeline — mode: {mode}")
    print(f"  Started  : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Manifest : {cfg.PAPERS_MANIFEST}")
    print(f"  PDFs dir : {cfg.PDFS_DIR}")

    # Show current KB state before starting
    stats = get_manifest_stats()
    print(f"\n  Current KB: {stats['total']} papers total, "
          f"{stats['fulltext_retrieved']} with full text\n")

    # ── Phase 1: PubMed ───────────────────────────────────────────────────────
    if mode in ("full", "update", "fetch"):
        # For update mode, only fetch papers from the last 7 days
        if mode == "update" and not date_filter:
            cutoff      = datetime.now() - timedelta(days=7)
            date_filter = f'"{cutoff.strftime("%Y/%m/%d")}"[PDAT] : "3000"[PDAT]'

        count = run_phase_1_pubmed(date_filter=date_filter, limit=limit)
        phase_results["PubMed fetch"] = f"{count} new papers"

    # ── Phase 2: Semantic Scholar ─────────────────────────────────────────────
    if mode in ("full", "update", "fetch") and not skip_s2:
        count = run_phase_2_semantic_scholar(limit=limit)
        phase_results["Semantic Scholar fetch"] = (
            f"{count} new papers" if cfg.USE_SEMANTIC_SCHOLAR else "pending key"
        )

    # ── Phase 3: PDF watcher ──────────────────────────────────────────────────
    if mode in ("full", "pdfs"):
        count = run_phase_3_pdf_watcher()
        phase_results["PDF watcher"] = f"{count} PDFs processed"

    # ── Phase 4: Full text retrieval ──────────────────────────────────────────
    if mode in ("full", "update"):
        counts = run_phase_4_fulltext(limit=limit)
        total  = sum(v for k, v in counts.items() if k != "skipped")
        phase_results["Full text retrieval"] = f"{total} papers retrieved"

    # ── Final report ──────────────────────────────────────────────────────────
    print_final_report(start_time, phase_results)


def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(
        description="pharma-lit-rag ingestion pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Full initial build:
    python ingestion/pipeline.py

  Weekly update (recent papers only):
    python ingestion/pipeline.py --update

  Process new university PDFs only:
    python ingestion/pipeline.py --pdfs

  Fetch metadata only (no full text):
    python ingestion/pipeline.py --fetch

  Test run (first 5 results per query):
    python ingestion/pipeline.py --limit 5

  Full build, skip Semantic Scholar:
    python ingestion/pipeline.py --no-s2
        """,
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--update", action="store_true",
        help="fetch only papers from last 7 days (used by weekly GitHub Action)"
    )
    mode_group.add_argument(
        "--pdfs", action="store_true",
        help="process local PDFs only (phases 3 only)"
    )
    mode_group.add_argument(
        "--fetch", action="store_true",
        help="fetch metadata only, skip full text retrieval (phases 1+2)"
    )

    parser.add_argument(
        "--limit", type=int, default=None,
        help="max results per query (useful for testing, e.g. --limit 5)"
    )
    parser.add_argument(
        "--no-s2", action="store_true",
        help="skip Semantic Scholar even if API key is configured"
    )

    return parser.parse_args()


# ── Run directly ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()

    if args.update:
        mode = "update"
    elif args.pdfs:
        mode = "pdfs"
    elif args.fetch:
        mode = "fetch"
    else:
        mode = "full"

    run_pipeline(
        mode    = mode,
        limit   = args.limit,
        skip_s2 = args.no_s2,
    )