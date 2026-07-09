"""
radlearn/database/client.py
────────────────────────────
Singleton clients for ChromaDB and SQLite.

Connection to system:
  - ChromaDB: stores vector embeddings for chunks and image captions.
    Collections are created lazily (get_or_create) so setup_database.py
    is the only place that needs to call these explicitly.
  - SQLite (via SQLAlchemy): stores all relational metadata.
    Engine is created once and reused across all database modules.

All other database modules import from here — never create their own clients.
"""

from __future__ import annotations

import chromadb
from chromadb.config import Settings
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session

from radlearn.config import CHROMA_DIR, SQLITE_PATH, CHUNKS_COLLECTION, IMAGES_COLLECTION


# ── ChromaDB ──────────────────────────────────────────────────────

_chroma_client: chromadb.PersistentClient | None = None


def get_chroma_client() -> chromadb.PersistentClient:
    """Return a cached ChromaDB persistent client."""
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
    return _chroma_client


def get_chunks_collection() -> chromadb.Collection:
    """
    Return the 'radlearn_chunks' ChromaDB collection.
    Configured for cosine similarity (hnsw:space=cosine).
    Cosine distance d → similarity = 1 - d.
    """
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=CHUNKS_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def get_images_collection() -> chromadb.Collection:
    """
    Return the 'radlearn_images' ChromaDB collection.
    Stores image caption embeddings for semantic image retrieval.
    """
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=IMAGES_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


# ── SQLite (SQLAlchemy) ───────────────────────────────────────────

_engine = None
_SessionFactory = None


def get_engine():
    """Return a cached SQLAlchemy engine connected to the SQLite database."""
    global _engine
    if _engine is None:
        _engine = create_engine(
            f"sqlite:///{SQLITE_PATH}",
            connect_args={"check_same_thread": False},
            echo=False,
        )
    return _engine


def get_session() -> Session:
    """Return a new SQLAlchemy session. Caller is responsible for closing it."""
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=get_engine(), autoflush=False)
    return _SessionFactory()


def execute_sql(sql: str, params: dict | None = None) -> list[dict]:
    """
    Execute a raw SQL statement and return results as a list of dicts.
    For INSERT/UPDATE/DELETE, returns an empty list.
    Commits automatically.

    Args:
        sql:    SQL string. Use :name style for named parameters.
        params: Dict of parameter values. Optional.

    Example:
        rows = execute_sql(
            "SELECT * FROM documents WHERE status = :s",
            {"s": "completed"}
        )
    """
    params = params or {}
    with get_engine().begin() as conn:
        result = conn.execute(text(sql), params)
        try:
            rows = result.fetchall()
            keys = result.keys()
            return [dict(zip(keys, row)) for row in rows]
        except Exception:
            return []
