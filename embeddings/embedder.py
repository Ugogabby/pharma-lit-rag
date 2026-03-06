"""
embeddings/embedder.py
───────────────────────
Converts chunks into vectors and stores them in Pinecone.

What this file does:
  1. Reads all_chunks.json produced by chunker.py
  2. Sends each chunk's text to Voyage AI → gets back a vector of 1024 numbers
  3. Upserts vectors + metadata into Pinecone serverless index
  4. Saves a local cache of embeddings to data/embeddings/ so we never
     re-embed a chunk we have already processed

Why batch processing?
  Voyage AI accepts up to 128 texts per request. Sending chunks in batches
  of 128 is 128x faster than sending one at a time and stays within rate limits.

Why a local cache?
  If the embedder crashes halfway through 150,000 chunks, you don't want to
  start over. The cache records which chunk_ids have been embedded. On restart
  it skips those and continues from where it left off.

Pinecone metadata limits:
  Pinecone stores metadata alongside each vector for filtering.
  Metadata values must be strings, numbers, or lists of strings.
  We store: pmid, section, year, journal, authors, citation, fulltext_source.
  The chunk text itself is NOT stored in Pinecone — only the vector and metadata.
  The full text is retrieved from all_chunks.json using the chunk_id.

Cost estimate for 150,000 chunks:
  Voyage AI voyage-3: ~$0.12 per 1M tokens, avg 150 tokens/chunk
  150,000 × 150 / 1,000,000 × $0.12 ≈ $2.70 total
  This is a one-time cost — cached forever after.
"""

import os
import json
import time
from pathlib import Path
from tqdm import tqdm

import voyageai
from pinecone import Pinecone, ServerlessSpec

from config import cfg


# ── Batching parameters ───────────────────────────────────────────────────────
VOYAGE_BATCH_SIZE  = 128   # Voyage AI max texts per request
PINECONE_BATCH_SIZE = 200  # Pinecone max vectors per upsert

# ── Embedding dimension for voyage-3 ─────────────────────────────────────────
EMBEDDING_DIM = 1024


def get_or_create_pinecone_index(pc: Pinecone) -> object:
    """
    Gets the Pinecone index if it exists, creates it if it doesn't.

    We use Pinecone Serverless — no fixed infrastructure, pay per query.
    The index is created once and reused forever.

    Serverless spec:
      cloud="aws", region="us-east-1" — US East is the default free region.
      If your Pinecone account is in a different region, update cfg.PINECONE_REGION.

    Args:
        pc: Pinecone client instance

    Returns:
        Pinecone index object ready for upsert and query
    """
    index_name = cfg.PINECONE_INDEX

    # Check if index already exists
    existing = [idx.name for idx in pc.list_indexes()]

    if index_name not in existing:
        print(f"Creating Pinecone index '{index_name}'...")
        pc.create_index(
            name      = index_name,
            dimension = EMBEDDING_DIM,
            metric    = "cosine",        # cosine similarity for text embeddings
            spec      = ServerlessSpec(
                cloud  = "aws",
                region = "us-east-1",    # change if your account uses a different region
            ),
        )
        # Wait for index to be ready
        print("  Waiting for index to be ready...")
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(2)
        print(f"  ✓ Index '{index_name}' created and ready")
    else:
        print(f"✓ Using existing Pinecone index '{index_name}'")

    return pc.Index(index_name)


def load_embedded_chunk_ids(cache_path: Path) -> set[str]:
    """
    Loads the set of chunk_ids already embedded from the local cache.
    Allows safe restart if the embedder is interrupted.

    Args:
        cache_path: path to the embedded_ids.json cache file

    Returns:
        set of chunk_id strings already processed
    """
    if not cache_path.exists():
        return set()

    with open(cache_path, "r", encoding="utf-8") as f:
        return set(json.load(f))


def save_embedded_chunk_ids(cache_path: Path, chunk_ids: set[str]):
    """
    Saves the updated set of embedded chunk_ids to the cache.

    Args:
        cache_path: path to the cache file
        chunk_ids:  complete set of all embedded chunk_ids so far
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(list(chunk_ids), f)


def build_pinecone_metadata(chunk: dict) -> dict:
    """
    Extracts the fields to store as Pinecone metadata.

    Pinecone metadata rules:
      - Values must be string, number, boolean, or list of strings
      - Total metadata per vector must be under 40KB
      - We do NOT store the full chunk text — only lookup fields

    The chunk text is retrieved from all_chunks.json using chunk_id
    when a search returns this vector.

    Args:
        chunk: chunk dict from all_chunks.json

    Returns:
        metadata dict safe for Pinecone storage
    """
    return {
        "chunk_id":        chunk["chunk_id"],
        "pmid":            str(chunk.get("pmid",    "")),
        "section":         chunk.get("section",     ""),
        "year":            str(chunk.get("year",    "")),
        "journal":         chunk.get("journal",     "")[:100],  # cap length
        "authors":         chunk.get("authors",     "")[:100],
        "title":           chunk.get("title",       "")[:200],
        "citation":        chunk.get("citation",    "")[:300],
        "fulltext_source": chunk.get("fulltext_source", ""),
        "doi":             chunk.get("doi",         ""),
    }


def embed_and_upsert_batch(
    chunks:     list[dict],
    voyage_client: voyageai.Client,
    index,
    embedded_ids: set[str],
    cache_path:   Path,
) -> int:
    """
    Embeds a batch of chunks with Voyage AI and upserts to Pinecone.

    Process:
      1. Extract text from each chunk
      2. Send to Voyage AI with input_type="document"
         (different from "query" — optimised for storage, not search)
      3. Pair each embedding with its metadata
      4. Upsert to Pinecone in batches of 200

    Args:
        chunks:        list of chunk dicts to embed
        voyage_client: Voyage AI client
        index:         Pinecone index object
        embedded_ids:  set of already-embedded chunk_ids (updated in place)
        cache_path:    path to save updated cache after each batch

    Returns:
        number of chunks successfully embedded
    """
    if not chunks:
        return 0

    texts = [c["text"] for c in chunks]

    try:
        # Embed with Voyage AI
        # input_type="document" optimises the embedding for storage/retrieval
        # (as opposed to input_type="query" which optimises for search queries)
        result     = voyage_client.embed(
            texts,
            model      = cfg.EMBED_MODEL,    # "voyage-3"
            input_type = "document",
        )
        embeddings = result.embeddings

    except Exception as e:
        print(f"\n  ⚠ Voyage AI embedding failed: {e}")
        return 0

    # ── Build Pinecone vectors ────────────────────────────────────────────────
    vectors = []
    for chunk, embedding in zip(chunks, embeddings):
        vectors.append({
            "id":       chunk["chunk_id"],      # unique ID in Pinecone
            "values":   embedding,              # the 1024-dimensional vector
            "metadata": build_pinecone_metadata(chunk),
        })

    # ── Upsert to Pinecone in sub-batches of 200 ─────────────────────────────
    upserted = 0
    for i in range(0, len(vectors), PINECONE_BATCH_SIZE):
        batch = vectors[i : i + PINECONE_BATCH_SIZE]
        try:
            index.upsert(vectors=batch)
            upserted += len(batch)
        except Exception as e:
            print(f"\n  ⚠ Pinecone upsert failed for batch {i}: {e}")
            continue

    # ── Update cache ──────────────────────────────────────────────────────────
    for chunk in chunks:
        embedded_ids.add(chunk["chunk_id"])
    save_embedded_chunk_ids(cache_path, embedded_ids)

    return upserted


def run_embedder(
    limit:        int  = None,
    no_context:   bool = False,
) -> int:
    """
    Main entry point — embeds all chunks and upserts to Pinecone.

    Steps:
      1. Load all chunks from all_chunks.json
      2. Filter out already-embedded chunks using cache
      3. Connect to Pinecone, create index if needed
      4. Process chunks in batches of 128 (Voyage AI limit)
      5. Print progress and final summary

    Args:
        limit:      process only first N chunks (for testing)
        no_context: unused here, kept for CLI consistency

    Returns:
        total number of chunks in Pinecone after this run
    """
    chunks_dir  = Path(cfg.CHUNKS_DIR)
    chunks_path = chunks_dir / "all_chunks.json"
    cache_path  = Path(cfg.EMBED_DIR) / "embedded_ids.json"

    # ── Load chunks ───────────────────────────────────────────────────────────
    if not chunks_path.exists():
        print(f"No chunks found at {chunks_path}")
        print("Run chunker.py first: python -m embeddings.chunker")
        return 0

    print("Loading chunks...")
    with open(chunks_path, "r", encoding="utf-8") as f:
        all_chunks = json.load(f)
    print(f"  Total chunks in KB : {len(all_chunks)}")

    # ── Filter already-embedded ───────────────────────────────────────────────
    embedded_ids = load_embedded_chunk_ids(cache_path)
    print(f"  Already embedded   : {len(embedded_ids)}")

    to_embed = [c for c in all_chunks if c["chunk_id"] not in embedded_ids]
    if limit:
        to_embed = to_embed[:limit]
    print(f"  To embed this run  : {len(to_embed)}\n")

    if not to_embed:
        print("All chunks already embedded. Pinecone is up to date.")
        return len(all_chunks)

    # ── Connect to Voyage AI ──────────────────────────────────────────────────
    print("Connecting to Voyage AI...")
    voyage_client = voyageai.Client(api_key=cfg.VOYAGE_API_KEY)
    print("  ✓ Voyage AI connected")

    # ── Connect to Pinecone ───────────────────────────────────────────────────
    print("Connecting to Pinecone...")
    pc    = Pinecone(api_key=cfg.PINECONE_API_KEY)
    index = get_or_create_pinecone_index(pc)
    print("  ✓ Pinecone connected\n")

    # ── Embed in batches ──────────────────────────────────────────────────────
    total_upserted = 0
    batches        = [
        to_embed[i : i + VOYAGE_BATCH_SIZE]
        for i in range(0, len(to_embed), VOYAGE_BATCH_SIZE)
    ]

    print(f"Embedding {len(to_embed)} chunks in {len(batches)} batches...")
    print(f"Estimated cost: ~${len(to_embed) * 150 / 1_000_000 * 0.12:.2f}\n")

    for batch in tqdm(batches, desc="Embedding + upserting"):
        upserted = embed_and_upsert_batch(
            batch, voyage_client, index, embedded_ids, cache_path
        )
        total_upserted += upserted

        # Brief pause to respect Voyage AI rate limits
        time.sleep(0.1)

    # ── Get final Pinecone stats ──────────────────────────────────────────────
    try:
        stats       = index.describe_index_stats()
        total_in_db = stats.get("total_vector_count", 0)
    except Exception:
        total_in_db = len(embedded_ids)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"  Embedding complete")
    print(f"  Chunks embedded this run : {total_upserted}")
    print(f"  Total vectors in Pinecone: {total_in_db}")
    print(f"  Local cache updated      : {cache_path}")
    print(f"{'='*50}\n")

    return total_in_db


# ── Run directly ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    limit = None
    if "--limit" in sys.argv:
        idx   = sys.argv.index("--limit")
        limit = int(sys.argv[idx + 1])

    run_embedder(limit=limit)