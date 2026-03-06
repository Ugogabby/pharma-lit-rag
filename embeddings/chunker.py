"""
embeddings/chunker.py
──────────────────────
Splits extracted full text into overlapping chunks ready for embedding.

What this file does:
  1. Reads every {PMID}_fulltext.json file from data/chunks/
  2. Splits each section into overlapping chunks of ~600 characters
  3. Labels every chunk with: PMID, section, authors, year, journal, citation
  4. Sends each chunk to Claude Haiku to add a context sentence
     (contextual retrieval — improves embedding quality significantly)
  5. Saves all chunks to data/chunks/all_chunks.json
  6. Builds and saves the BM25 index to data/chunks/bm25_index.json

Why overlapping chunks?
  If a key sentence falls at the boundary between two chunks, overlap
  ensures it appears fully in at least one chunk. 100-char overlap means
  the last 100 characters of chunk N are repeated at the start of chunk N+1.

Why contextual enrichment?
  A raw chunk from a methods section might say:
    "We used a 365-day washout period prior to the index date."
  Without context, the embedding doesn't know this is about gabapentinoids,
  or which study it's from. Claude Haiku prepends:
    "From Gomes et al. (2017) BMJ, a study of gabapentinoid-opioid
     co-prescription and mortality: We used a 365-day washout period..."
  Now the embedding carries both content AND context. Retrieval improves
  dramatically for specific methodological queries.

Cost estimate:
  ~150,000 chunks × Claude Haiku pricing ≈ $1.50 total
  This is a one-time cost — chunks are cached to disk.
"""

import os
import json
import time
import re
from pathlib import Path
from tqdm import tqdm
import anthropic

from config import cfg


# ── Chunking parameters ───────────────────────────────────────────────────────
CHUNK_SIZE    = cfg.CHUNK_SIZE     # 600 characters per chunk
CHUNK_OVERLAP = cfg.CHUNK_OVERLAP  # 100 characters overlap between chunks

# ── Section priority order ────────────────────────────────────────────────────
# Methods and Results sections are highest value for pharmacoepi queries.
# We process them first so if we hit any limits, we have the best content.
SECTION_PRIORITY = [
    "methods",
    "results",
    "discussion",
    "conclusion",
    "introduction",
    "abstract",
    "other",
]

# ── Minimum chunk length ──────────────────────────────────────────────────────
# Chunks shorter than this are too small to be useful — skip them.
MIN_CHUNK_LENGTH = 100


def load_paper_metadata(manifest_path: str) -> dict[str, dict]:
    """
    Loads all paper metadata from papers.jsonl into a dict keyed by PMID.
    Used to attach citation info to every chunk.

    Args:
        manifest_path: path to papers.jsonl

    Returns:
        dict mapping PMID string → paper metadata dict
    """
    metadata = {}
    path = Path(manifest_path)
    if not path.exists():
        return metadata

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                paper = json.loads(line)
                pmid  = str(paper.get("pmid", ""))
                if pmid:
                    metadata[pmid] = paper
            except json.JSONDecodeError:
                continue

    return metadata


def split_into_chunks(text: str, chunk_size: int = CHUNK_SIZE,
                      overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Splits a text string into overlapping chunks.

    Algorithm:
      - Try to split at sentence boundaries (. ! ?) to avoid cutting mid-sentence
      - If no sentence boundary found within the window, split at word boundary
      - Each chunk starts overlap characters before the previous chunk ended

    Args:
        text:       the text to split
        chunk_size: target characters per chunk
        overlap:    characters to repeat between consecutive chunks

    Returns:
        list of chunk strings
    """
    if not text or len(text) < MIN_CHUNK_LENGTH:
        return []

    # Clean up excessive whitespace
    text = " ".join(text.split())

    chunks = []
    start  = 0

    while start < len(text):
        end = start + chunk_size

        if end >= len(text):
            # Last chunk — take everything remaining
            chunk = text[start:].strip()
            if len(chunk) >= MIN_CHUNK_LENGTH:
                chunks.append(chunk)
            break

        # Try to find a sentence boundary near the end of this chunk
        # Look for . ! ? followed by a space within the last 150 chars
        boundary = -1
        search_start = max(start + chunk_size - 150, start)
        for match in re.finditer(r'[.!?]\s', text[search_start:end + 50]):
            boundary = search_start + match.end()

        if boundary > start + MIN_CHUNK_LENGTH:
            # Found a sentence boundary — split there
            chunk = text[start:boundary].strip()
        else:
            # No sentence boundary — split at word boundary
            space_pos = text.rfind(' ', start, end)
            if space_pos > start:
                chunk = text[start:space_pos].strip()
                boundary = space_pos + 1
            else:
                chunk = text[start:end].strip()
                boundary = end

        if len(chunk) >= MIN_CHUNK_LENGTH:
            chunks.append(chunk)

        # Next chunk starts overlap characters before current end
        start = max(boundary - overlap, start + 1)

    return chunks


def generate_context(chunk_text: str, paper: dict,
                     section: str, client: anthropic.Anthropic) -> str:
    """
    Uses Claude Haiku to generate a context sentence for a chunk.

    This is the contextual retrieval step. It prepends a brief description
    of what concept the chunk addresses and which paper it comes from.
    This dramatically improves retrieval quality for specific queries.

    Args:
        chunk_text: the raw chunk text
        paper:      paper metadata dict (title, authors, year, journal)
        section:    which section this chunk came from
        client:     Anthropic client instance

    Returns:
        context-enriched chunk text: "{context sentence} {original chunk}"
    """
    # Build a compact paper reference for the prompt
    paper_ref = (
        f"{paper.get('authors', 'Unknown')[:50]} "
        f"({paper.get('year', '????')}), "
        f"{paper.get('journal', 'Unknown journal')[:40]}"
    )

    prompt = (
        f"Paper: {paper_ref}\n"
        f"Section: {section}\n"
        f"Chunk: {chunk_text[:500]}\n\n"
        f"Write ONE sentence (max 30 words) describing what specific concept, "
        f"method, finding, or result this chunk addresses and which paper it is from. "
        f"Be specific — mention drug names, methods, or outcomes if present. "
        f"Reply with only that sentence, no preamble."
    )

    try:
        response = client.messages.create(
            model      = "claude-haiku-4-5",
            max_tokens = 80,
            messages   = [{"role": "user", "content": prompt}],
        )
        context = response.content[0].text.strip()
        # Prepend context to chunk
        return f"{context} {chunk_text}"

    except Exception:
        # If context generation fails, return chunk unchanged
        return chunk_text


def process_paper(pmid: str, fulltext: dict, paper_meta: dict,
                  client: anthropic.Anthropic,
                  use_context: bool = True) -> list[dict]:
    """
    Converts one paper's full text into a list of chunk dicts.

    For each section in the full text:
      1. Split into overlapping chunks
      2. Optionally enrich each chunk with AI-generated context
      3. Attach all metadata: PMID, section, citation, year, journal

    Args:
        pmid:        PubMed ID
        fulltext:    dict with sections dict and source string
        paper_meta:  paper metadata from papers.jsonl
        client:      Anthropic client for context generation
        use_context: whether to run contextual enrichment (costs API calls)

    Returns:
        list of chunk dicts ready for embedding
    """
    sections = fulltext.get("sections", {})
    source   = fulltext.get("source", "unknown")
    chunks   = []
    chunk_id = 0

    for section in SECTION_PRIORITY:
        text = sections.get(section, "").strip()
        if not text:
            continue

        # Split section text into overlapping chunks
        text_chunks = split_into_chunks(text)

        for chunk_text in text_chunks:
            # Optionally enrich with context sentence
            if use_context and client:
                enriched = generate_context(chunk_text, paper_meta,
                                            section, client)
                # Small delay to respect Haiku rate limits
                time.sleep(0.05)
            else:
                enriched = chunk_text

            # Build the chunk dict — everything needed for retrieval
            chunk = {
                # Unique identifier for this chunk
                "chunk_id": f"{pmid}_{section}_{chunk_id}",

                # The text that gets embedded
                "text": enriched,

                # Original text without context (for display)
                "text_original": chunk_text,

                # Paper identifiers
                "pmid":    pmid,
                "doi":     paper_meta.get("doi",  ""),
                "pmc_id":  paper_meta.get("pmc_id", ""),

                # Section this chunk came from
                # Critical for metadata filtering in Pinecone
                "section": section,

                # Bibliographic metadata — shown in retrieval results
                "title":   paper_meta.get("title",   ""),
                "authors": paper_meta.get("authors", ""),
                "year":    paper_meta.get("year",    ""),
                "journal": paper_meta.get("journal", ""),
                "citation":paper_meta.get("citation",""),

                # Source quality indicator
                # pmc > unpaywall > local_pdf > abstract_only
                "fulltext_source": source,

                # Position within paper (useful for ordering results)
                "chunk_index": chunk_id,
            }

            chunks.append(chunk)
            chunk_id += 1

    return chunks


def run_chunker(
    use_context:  bool = True,
    limit:        int  = None,
    skip_existing: bool = True,
) -> int:
    """
    Main entry point — chunks all papers in data/chunks/*_fulltext.json

    Args:
        use_context:   run contextual enrichment via Claude Haiku
        limit:         process only first N papers (for testing)
        skip_existing: skip papers already in all_chunks.json

    Returns:
        total number of chunks created
    """
    chunks_dir   = Path(cfg.CHUNKS_DIR)
    output_path  = chunks_dir / "all_chunks.json"
    bm25_path    = chunks_dir / "bm25_corpus.json"

    # ── Load paper metadata ───────────────────────────────────────────────────
    print("Loading paper metadata...")
    paper_metadata = load_paper_metadata(cfg.PAPERS_MANIFEST)
    print(f"  Loaded metadata for {len(paper_metadata)} papers")

    # ── Find fulltext files to process ────────────────────────────────────────
    fulltext_files = sorted(chunks_dir.glob("*_fulltext.json"))
    if limit:
        fulltext_files = fulltext_files[:limit]

    print(f"  Found {len(fulltext_files)} fulltext files to chunk")

    # ── Load existing chunks to support incremental runs ──────────────────────
    existing_chunks  = []
    existing_pmids   = set()

    if skip_existing and output_path.exists():
        print("  Loading existing chunks...")
        with open(output_path, "r", encoding="utf-8") as f:
            existing_chunks = json.load(f)
        existing_pmids = {c["pmid"] for c in existing_chunks}
        print(f"  {len(existing_pmids)} papers already chunked — skipping")

    # Filter to only unprocessed files
    to_process = [
        f for f in fulltext_files
        if f.stem.replace("_fulltext", "") not in existing_pmids
    ]
    print(f"  Papers to chunk this run: {len(to_process)}\n")

    if not to_process:
        print("All papers already chunked.")
        return len(existing_chunks)

    # ── Set up Anthropic client for context generation ────────────────────────
    client = None
    if use_context:
        client = anthropic.Anthropic(api_key=cfg.ANTHROPIC_API_KEY)
        print(f"  Contextual enrichment: ON (Claude Haiku)")
    else:
        print(f"  Contextual enrichment: OFF (faster, lower quality)")

    # ── Process each paper ────────────────────────────────────────────────────
    all_new_chunks = []
    errors         = []

    for fulltext_file in tqdm(to_process, desc="Chunking papers"):
        pmid = fulltext_file.stem.replace("_fulltext", "")

        # Get paper metadata — use empty dict if not found
        paper_meta = paper_metadata.get(pmid, {
            "pmid": pmid, "title": "", "authors": "",
            "year": "", "journal": "", "citation": "",
        })

        try:
            with open(fulltext_file, "r", encoding="utf-8") as f:
                fulltext = json.load(f)

            chunks = process_paper(pmid, fulltext, paper_meta,
                                   client, use_context)
            all_new_chunks.extend(chunks)

        except Exception as e:
            errors.append({"pmid": pmid, "error": str(e)})
            continue

    # ── Save all chunks ───────────────────────────────────────────────────────
    all_chunks = existing_chunks + all_new_chunks
    chunks_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving {len(all_chunks)} total chunks...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False)

    # ── Build BM25 corpus ─────────────────────────────────────────────────────
    # Save just the text and chunk_id for BM25 indexing
    # The retriever builds the actual BM25 index from this at query time
    print("Saving BM25 corpus...")
    bm25_corpus = [
        {"chunk_id": c["chunk_id"], "text": c["text_original"]}
        for c in all_chunks
    ]
    with open(bm25_path, "w", encoding="utf-8") as f:
        json.dump(bm25_corpus, f, ensure_ascii=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    section_counts = {}
    for c in all_new_chunks:
        s = c["section"]
        section_counts[s] = section_counts.get(s, 0) + 1

    print(f"\n{'='*50}")
    print(f"  Chunking complete")
    print(f"  New chunks created      : {len(all_new_chunks)}")
    print(f"  Total chunks in KB      : {len(all_chunks)}")
    print(f"  Errors                  : {len(errors)}")
    print(f"  Output                  : {output_path}")
    print(f"\n  Chunks by section:")
    for section, count in sorted(section_counts.items(),
                                  key=lambda x: -x[1]):
        print(f"    {section:<20} : {count:>6}")
    print(f"{'='*50}\n")

    return len(all_chunks)


# ── Run directly ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    # Pass --no-context to skip contextual enrichment (faster, for testing)
    # Pass --limit N to process only first N papers
    use_context = "--no-context" not in sys.argv
    limit       = None
    if "--limit" in sys.argv:
        idx   = sys.argv.index("--limit")
        limit = int(sys.argv[idx + 1])

    run_chunker(use_context=use_context, limit=limit)