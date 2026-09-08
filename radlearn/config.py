"""
radlearn/config.py
──────────────────
Central configuration for the RadLearn application.
All paths, constants, and environment variables are resolved here.
Every other module imports from this file — never from os.environ directly.

Connection to system:
  - Reads from .env (local dev) or Streamlit secrets (deployed)
  - Creates all data directories on first import
  - Used by: database/client.py, ingestion/*, retrieval/*, chat/*
"""

import os
from pathlib import Path

# ── Load .env (local development only) ───────────────────────────
# In production (Streamlit Cloud), secrets come from st.secrets.
# We do a soft import so this file works even when streamlit isn't running
# (e.g. during setup_database.py).
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass


def _get(key: str, default: str = "") -> str:
    """Read from env first, then Streamlit secrets, then default."""
    value = os.getenv(key)
    if value:
        return value
    try:
        import streamlit as st
        return st.secrets.get(key, default)
    except Exception:
        return default


# ── Project root ──────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent  # /path/to/radlearn-chatbot

# ── Data directory (all persistent storage lives here) ────────────
DATA_DIR = Path(_get("DATA_DIR", "./data"))
if not DATA_DIR.is_absolute():
    DATA_DIR = BASE_DIR / DATA_DIR

CHROMA_DIR = DATA_DIR / "chroma"        # ChromaDB persistence
SQLITE_PATH = DATA_DIR / "radlearn.db"  # SQLite relational DB
DOCUMENTS_DIR = DATA_DIR / "documents"  # Uploaded PDFs / DOCX
IMAGES_DIR = DATA_DIR / "images"        # Extracted PDF figures

# Create directories on import (safe to call multiple times)
for _d in [DATA_DIR, CHROMA_DIR, DOCUMENTS_DIR, IMAGES_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── API Keys ──────────────────────────────────────────────────────
GOOGLE_API_KEY: str = _get("GOOGLE_API_KEY")
GROQ_API_KEY: str = _get("GROQ_API_KEY")
OPENAI_API_KEY: str = _get("OPENAI_API_KEY")
ANTHROPIC_API_KEY: str = _get("ANTHROPIC_API_KEY")
ADMIN_PASSWORD: str = _get("ADMIN_PASSWORD", "radlearn-admin")

# ── ChromaDB collection names ─────────────────────────────────────
CHUNKS_COLLECTION: str = "radlearn_chunks"
IMAGES_COLLECTION: str = "radlearn_images"

# ── Chunking parameters ───────────────────────────────────────────
CHUNK_SIZE: int = 512        # Target tokens per chunk
CHUNK_OVERLAP: int = 64      # Overlap tokens between adjacent chunks
MIN_CHUNK_WORDS: int = 10    # Skip chunks with fewer words than this

# ── Embedding ─────────────────────────────────────────────────────
# TODO: Migrate to local embeddings (e.g., sentence-transformers) to reduce cloud API dependencies.
EMBEDDING_MODEL: str = "models/gemini-embedding-001"
EMBEDDING_DIM: int = 3072
EMBEDDING_BATCH_SIZE: int = 20    # Texts per API call (free tier: 100 req/min)
EMBEDDING_MAX_TOKENS: int = 2000  # Truncate input if longer

# ── Retrieval parameters ──────────────────────────────────────────
MIN_SIMILARITY_THRESHOLD: float = 0.60  # Below this → "not found"
SEMANTIC_TOP_K: int = 15
KEYWORD_TOP_K: int = 15
CONTEXT_CHUNKS: int = 5    # Chunks sent to LLM
IMAGE_TOP_K: int = 3       # Images returned per answer

# ── LLM ──────────────────────────────────────────────────────────
LLM_PROVIDER: str = _get("LLM_PROVIDER", "groq").lower()

def _resolve_default_model(provider: str) -> str:
    if provider == "groq":
        return "openai/gpt-oss-120b"
    elif provider == "gemini":
        return "gemini-2.5-flash"
    elif provider == "openai":
        return "gpt-4o-mini"
    elif provider == "anthropic":
        return "claude-3-5-sonnet-20241022"
    return "openai/gpt-oss-120b"

LLM_MODEL: str = _get("LLM_MODEL", _resolve_default_model(LLM_PROVIDER))
LLM_TEMPERATURE: float = float(_get("LLM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS: int = int(_get("LLM_MAX_TOKENS", "1024"))




# ── Document ingestion limits ─────────────────────────────────────
MAX_FILE_SIZE_BYTES: int = 50 * 1024 * 1024   # 50 MB
MIN_CONTENT_WORDS: int = 200                  # Reject web pages below this
MAX_INGEST_RETRIES: int = 3

# ── Image extraction filters ─────────────────────────────────────
IMG_MIN_WIDTH: int = 150    # px — skip smaller images
IMG_MIN_HEIGHT: int = 150
IMG_MAX_WIDTH: int = 4000
IMG_MAX_HEIGHT: int = 4000
IMG_MIN_SIZE_BYTES: int = 5_000  # 5 KB
