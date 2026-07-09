"""
radlearn/database/citations.py
───────────────────────────────
CRUD for the citations table.

Connection to system:
  - chat/citation_parser.py creates citation objects from the LLM answer.
  - chat/engine.py calls batch_insert_citations() after saving the assistant message.
  - The citations table is the audit trail linking every answer claim to its source.
  - chunk_id is a ChromaDB document ID (not a SQLite FK) — stored as TEXT.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from radlearn.database.client import execute_sql


def batch_insert_citations(citations: list[dict], message_id: str) -> None:
    """
    Persist all citations for a single assistant message.

    Args:
        citations:  List of citation dicts produced by citation_parser.py.
                    Each dict must contain:
                        citation_number, chunk_id, document_id,
                        cited_text, relevance_score, rank_position,
                        formatted_citation, source_url, page_number
        message_id: UUID of the assistant message these citations belong to.
    """
    for c in citations:
        execute_sql(
            """
            INSERT INTO citations
                (id, message_id, chunk_id, document_id, citation_number,
                 cited_text, relevance_score, rank_position,
                 formatted_citation, source_url, page_number, created_at)
            VALUES
                (:id, :message_id, :chunk_id, :document_id, :citation_number,
                 :cited_text, :relevance_score, :rank_position,
                 :formatted_citation, :source_url, :page_number, :created_at)
            """,
            {
                "id":                 str(uuid.uuid4()),
                "message_id":         message_id,
                "chunk_id":           c["chunk_id"],
                "document_id":        c["document_id"],
                "citation_number":    int(c["citation_number"]),
                "cited_text":         str(c.get("cited_text", ""))[:500],
                "relevance_score":    float(c.get("relevance_score", 0.0)),
                "rank_position":      int(c.get("rank_position", 0)),
                "formatted_citation": str(c.get("formatted_citation", "")),
                "source_url":         str(c.get("source_url", "")),
                "page_number":        int(c.get("page_number") or -1),
                "created_at":         _now(),
            },
        )


def get_citations_for_message(message_id: str) -> list[dict]:
    """
    Return all citations for an assistant message, ordered by citation number.
    Used by the UI to render citation cards.
    """
    return execute_sql(
        """
        SELECT c.*, d.title AS doc_title, d.doc_type, d.author AS doc_author,
               d.publication_year AS doc_year, d.specialty AS doc_specialty
        FROM citations c
        JOIN documents d ON c.document_id = d.id
        WHERE c.message_id = :id
        ORDER BY c.citation_number ASC
        """,
        {"id": message_id},
    )


def get_citation_stats() -> dict:
    """Return aggregate citation statistics for the admin statistics page."""
    rows = execute_sql(
        """
        SELECT
            COUNT(*)                                           AS total_citations,
            AVG(relevance_score)                              AS avg_relevance,
            COUNT(DISTINCT document_id)                        AS unique_docs_cited
        FROM citations
        """
    )
    return rows[0] if rows else {}


# ── Helpers ───────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
