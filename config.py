"""
config.py
─────────
Central configuration for the entire project.
Every other file imports settings from here.
API keys are read from the .env file — never hardcoded in code.
"""

import os
from dotenv import load_dotenv

# Reads your .env file and makes every line available via os.getenv()
load_dotenv()

class Config:

    # ── API Keys ──────────────────────────────────────────────────────────────
    ANTHROPIC_API_KEY         = os.getenv("ANTHROPIC_API_KEY")
    VOYAGE_API_KEY            = os.getenv("VOYAGE_API_KEY")
    PINECONE_API_KEY          = os.getenv("PINECONE_API_KEY")
    COHERE_API_KEY            = os.getenv("COHERE_API_KEY", "")
    NCBI_API_KEY              = os.getenv("NCBI_API_KEY", "")
    SEMANTIC_SCHOLAR_API_KEY  = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
    UNPAYWALL_EMAIL           = os.getenv("UNPAYWALL_EMAIL", "")

    # ── Pinecone ──────────────────────────────────────────────────────────────
    PINECONE_INDEX            = os.getenv("PINECONE_INDEX_NAME", "pharma-lit-rag")
    PINECONE_REGION           = os.getenv("PINECONE_ENVIRONMENT", "us-east-1-aws")

    # ── Models ────────────────────────────────────────────────────────────────
    EMBED_MODEL               = os.getenv("EMBEDDING_MODEL", "voyage-3")
    LLM_MODEL                 = os.getenv("LLM_MODEL", "claude-sonnet-4-6")

    # ── Chunking ──────────────────────────────────────────────────────────────
    # CHUNK_SIZE: characters per chunk (~100 words at 600 chars)
    # CHUNK_OVERLAP: characters carried over between chunks for context continuity
    # TOP_K: number of chunks returned per query after reranking
    CHUNK_SIZE                = int(os.getenv("CHUNK_SIZE", 600))
    CHUNK_OVERLAP             = int(os.getenv("CHUNK_OVERLAP", 100))
    TOP_K                     = int(os.getenv("TOP_K_RETRIEVAL", 8))

    # ── Retrieval ─────────────────────────────────────────────────────────────
    # BM25_CANDIDATES: how many results BM25 returns before RRF merging
    # SEMANTIC_CANDIDATES: how many results Pinecone returns before RRF merging
    # RERANK_CANDIDATES: how many merged results go to Cohere reranker
    BM25_CANDIDATES           = int(os.getenv("BM25_CANDIDATES", 20))
    SEMANTIC_CANDIDATES       = int(os.getenv("SEMANTIC_CANDIDATES", 20))
    RERANK_CANDIDATES         = int(os.getenv("RERANK_CANDIDATES", 20))

    # ── File Paths ────────────────────────────────────────────────────────────
    PAPERS_MANIFEST           = "data/papers.jsonl"   # one paper per line, tracks all KB papers
    CHUNKS_DIR                = "data/chunks"          # chunked text saved as JSON
    EMBED_DIR                 = "data/embeddings"      # local backup of vectors
    PDFS_DIR                  = "data/pdfs"            # university-downloaded PDFs go here
    RAW_DIR                   = "data/raw"             # raw API responses

    # ── Ingestion ─────────────────────────────────────────────────────────────
    # MAX_RESULTS_PER_QUERY: PubMed results per search query
    # FETCH_BATCH_SIZE: how many PMIDs to fetch metadata for in one API call
    MAX_RESULTS_PER_QUERY     = int(os.getenv("MAX_RESULTS_PER_QUERY", 100))
    FETCH_BATCH_SIZE          = 200

    # ── Feature Flags ─────────────────────────────────────────────────────────
    # Set to "true" in .env to enable each source
    # Semantic Scholar activates automatically when key is not "pending"
    USE_SEMANTIC_SCHOLAR      = (
        SEMANTIC_SCHOLAR_API_KEY not in ("", "pending")
    )
    USE_UNPAYWALL             = bool(UNPAYWALL_EMAIL)
    USE_COHERE_RERANK         = bool(COHERE_API_KEY)


# Single shared instance imported everywhere
# Usage in any file: from config import cfg
cfg = Config()


# ── Startup validation ────────────────────────────────────────────────────────
# Run this block when executing config.py directly to verify your setup
if __name__ == "__main__":
    print("=== Config Validation ===")
    print(f"Anthropic API key    : {'✓' if cfg.ANTHROPIC_API_KEY  else '✗ MISSING'}")
    print(f"Voyage AI key        : {'✓' if cfg.VOYAGE_API_KEY     else '✗ MISSING'}")
    print(f"Pinecone key         : {'✓' if cfg.PINECONE_API_KEY   else '✗ MISSING'}")
    print(f"Cohere key           : {'✓' if cfg.COHERE_API_KEY     else '○ optional'}")
    print(f"NCBI key             : {'✓' if cfg.NCBI_API_KEY       else '○ optional'}")
    print(f"Semantic Scholar key : {'✓' if cfg.USE_SEMANTIC_SCHOLAR else '○ pending'}")
    print(f"Unpaywall email      : {'✓' if cfg.USE_UNPAYWALL      else '○ not set'}")
    print()
    print(f"Semantic Scholar active : {cfg.USE_SEMANTIC_SCHOLAR}")
    print(f"Unpaywall active        : {cfg.USE_UNPAYWALL}")
    print(f"Cohere rerank active    : {cfg.USE_COHERE_RERANK}")