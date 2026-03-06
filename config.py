"""
config.py
─────────
Central configuration for the entire project.
Every other file imports settings from here.
API keys are read from the .env file — never hardcoded in code.
"""

import os
from dotenv import load_dotenv

# This reads your .env file and makes every line available via os.getenv()
load_dotenv()

class Config:

    # ── API Keys ──────────────────────────────────────────────────────────────
    ANTHROPIC_API_KEY  = os.getenv("ANTHROPIC_API_KEY")
    VOYAGE_API_KEY     = os.getenv("VOYAGE_API_KEY")
    PINECONE_API_KEY   = os.getenv("PINECONE_API_KEY")
    COHERE_API_KEY     = os.getenv("COHERE_API_KEY", "")
    NCBI_API_KEY       = os.getenv("NCBI_API_KEY", "")

    # ── Pinecone ──────────────────────────────────────────────────────────────
    PINECONE_INDEX     = os.getenv("PINECONE_INDEX_NAME", "pharma-lit-rag")
    PINECONE_REGION    = os.getenv("PINECONE_ENVIRONMENT", "us-east-1-aws")

    # ── Models ────────────────────────────────────────────────────────────────
    EMBED_MODEL        = os.getenv("EMBEDDING_MODEL", "voyage-3")
    LLM_MODEL          = os.getenv("LLM_MODEL", "claude-sonnet-4-5")

    # ── Chunking ──────────────────────────────────────────────────────────────
    CHUNK_SIZE         = int(os.getenv("CHUNK_SIZE", 600))
    CHUNK_OVERLAP      = int(os.getenv("CHUNK_OVERLAP", 100))
    TOP_K              = int(os.getenv("TOP_K_RETRIEVAL", 8))

    # ── File Paths ────────────────────────────────────────────────────────────
    PAPERS_MANIFEST    = "data/papers.jsonl"
    CHUNKS_DIR         = "data/chunks"
    EMBED_DIR          = "data/embeddings"
    PDFS_DIR           = "data/pdfs"


# One shared instance imported everywhere
# Usage in any file: from config import cfg
cfg = Config()