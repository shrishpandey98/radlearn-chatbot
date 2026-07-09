"""
radlearn/database/documents.py
───────────────────────────────
CRUD operations for knowledge_sources and documents tables.

Connection to system:
  - Called by ingestion/pipeline.py when a new document is submitted.
  - Called by the Admin dashboard to list, filter, and delete documents.
  - knowledge_sources groups related documents (e.g. "ACR Guidelines").
  - documents.content_hash prevents duplicate ingestion.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from radlearn.database.client import execute_sql


# ── knowledge_sources ─────────────────────────────────────────────

def create_knowledge_source(data: dict) -> dict:
    """
    Create a new knowledge source record.

    Args:
        data: dict with keys:
            name         (required)
            source_type  optional, default 'manual_upload'
            base_url     optional
            description  optional
            license_type optional, default 'unknown'
            specialty_tags optional, list → stored as JSON string

    Returns: the created record as a dict.
    """
    import json
    record = {
        "id":            str(uuid.uuid4()),
        "name":          data["name"],
        "source_type":   data.get("source_type", "manual_upload"),
        "base_url":      data.get("base_url", ""),
        "description":   data.get("description", ""),
        "license_type":  data.get("license_type", "unknown"),
        "specialty_tags": json.dumps(data.get("specialty_tags", [])),
        "is_active":     1,
        "created_at":    _now(),
        "updated_at":    _now(),
    }
    execute_sql(
        """
        INSERT INTO knowledge_sources
            (id, name, source_type, base_url, description,
             license_type, specialty_tags, is_active, created_at, updated_at)
        VALUES
            (:id, :name, :source_type, :base_url, :description,
             :license_type, :specialty_tags, :is_active, :created_at, :updated_at)
        """,
        record,
    )
    return record


def list_knowledge_sources(active_only: bool = False) -> list[dict]:
    """Return all knowledge sources, optionally filtered to active only."""
    sql = "SELECT * FROM knowledge_sources"
    if active_only:
        sql += " WHERE is_active = 1"
    sql += " ORDER BY name"
    return execute_sql(sql)


def get_knowledge_source(source_id: str) -> dict | None:
    rows = execute_sql(
        "SELECT * FROM knowledge_sources WHERE id = :id",
        {"id": source_id},
    )
    return rows[0] if rows else None


def get_or_create_default_source() -> dict:
    """
    Return the 'Manual Uploads' knowledge source, creating it if necessary.
    Used as the default when no source is specified during upload.
    """
    rows = execute_sql(
        "SELECT * FROM knowledge_sources WHERE source_type = 'manual_upload' LIMIT 1"
    )
    if rows:
        return rows[0]
    return create_knowledge_source({
        "name": "Manual Uploads",
        "source_type": "manual_upload",
        "license_type": "unknown",
    })


# ── documents ─────────────────────────────────────────────────────

def create_document(data: dict) -> dict:
    """
    Insert a new document record with status='pending'.

    Args:
        data: dict with keys:
            title           (required)
            doc_type        required: guideline|textbook|article|case_report|website|review|atlas
            specialty       required: general|neuro|breast|chest|msk|abdominal|...
            license_type    required
            knowledge_source_id  optional
            author          optional
            publication_year optional
            source_url      optional (web docs)
            file_path       optional (PDF/DOCX docs)
            content_hash    optional (SHA-256 of file bytes)
            citation_format optional

    Returns: the created document record as a dict.
    """
    record = {
        "id":                   str(uuid.uuid4()),
        "knowledge_source_id":  data.get("knowledge_source_id", ""),
        "title":                data["title"],
        "author":               data.get("author", ""),
        "publication_year":     data.get("publication_year", 0),
        "source_url":           data.get("source_url", ""),
        "file_path":            data.get("file_path", ""),
        "doc_type":             data["doc_type"],
        "specialty":            data.get("specialty", "general"),
        "subspecialty_tags":    "[]",
        "license_type":         data.get("license_type", "unknown"),
        "citation_format":      data.get("citation_format", ""),
        "status":               "pending",
        "content_hash":         data.get("content_hash", ""),
        "is_visible":           1,
        "total_pages":          0,
        "total_words":          0,
        "chunk_count":          0,
        "image_count":          0,
        "ingested_at":          _now(),
        "updated_at":           _now(),
        "project_id":           data.get("project_id", None),
    }
    execute_sql(
        """
        INSERT INTO documents
            (id, knowledge_source_id, title, author, publication_year,
             source_url, file_path, doc_type, specialty, subspecialty_tags,
             license_type, citation_format, status, content_hash,
             is_visible, total_pages, total_words, chunk_count, image_count,
             ingested_at, updated_at, project_id)
        VALUES
            (:id, :knowledge_source_id, :title, :author, :publication_year,
             :source_url, :file_path, :doc_type, :specialty, :subspecialty_tags,
             :license_type, :citation_format, :status, :content_hash,
             :is_visible, :total_pages, :total_words, :chunk_count, :image_count,
             :ingested_at, :updated_at, :project_id)
        """,
        record,
    )
    return record


def get_document(doc_id: str) -> dict | None:
    """Fetch a single document by ID. Returns None if not found."""
    rows = execute_sql(
        "SELECT * FROM documents WHERE id = :id",
        {"id": doc_id},
    )
    return rows[0] if rows else None


def list_documents(
    specialty: str | None = None,
    status: str | None = None,
    doc_type: str | None = None,
    visible_only: bool = False,
    project_id: str | None = None,
) -> list[dict]:
    """
    Return documents with optional filters.
    Results ordered by ingestion date descending (newest first).
    """
    clauses = []
    params: dict = {}

    if specialty:
        clauses.append("specialty = :specialty")
        params["specialty"] = specialty
    if status:
        clauses.append("status = :status")
        params["status"] = status
    if doc_type:
        clauses.append("doc_type = :doc_type")
        params["doc_type"] = doc_type
    if visible_only:
        clauses.append("is_visible = 1")
    if project_id:
        clauses.append("project_id = :project_id")
        params["project_id"] = project_id

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    sql = f"SELECT * FROM documents {where} ORDER BY ingested_at DESC"
    return execute_sql(sql, params)


def update_document(doc_id: str, updates: dict) -> None:
    """
    Partial update of a document record.

    Args:
        doc_id:  UUID of the document to update.
        updates: Dict of column → value pairs to update.
                 'updated_at' is always set automatically.
    """
    updates["updated_at"] = _now()
    updates["id"] = doc_id
    set_clauses = ", ".join(
        f"{k} = :{k}" for k in updates if k != "id"
    )
    execute_sql(
        f"UPDATE documents SET {set_clauses} WHERE id = :id",
        updates,
    )


def delete_document_record(doc_id: str) -> None:
    """
    Delete a document and all cascade-dependent records (jobs, messages→citations).
    Note: chunks are deleted from ChromaDB separately in pipeline.py.
    """
    execute_sql("DELETE FROM documents WHERE id = :id", {"id": doc_id})


def check_duplicate_hash(content_hash: str) -> dict | None:
    """
    Check whether a document with this hash already exists in the DB.

    Returns:
        Dict with 'id' and 'status' if a record exists (any status), else None.
        The caller can decide whether to skip (completed) or retry (failed).
    """
    if not content_hash:
        return None
    rows = execute_sql(
        "SELECT id, status FROM documents WHERE content_hash = :h",
        {"h": content_hash},
    )
    return rows[0] if rows else None


def get_document_stats() -> dict:
    """Return aggregate statistics for the knowledge base statistics page."""
    rows = execute_sql(
        """
        SELECT
            COUNT(*)                                     AS total_docs,
            SUM(CASE WHEN status='completed' THEN 1 ELSE 0 END) AS completed,
            SUM(CASE WHEN status='failed'    THEN 1 ELSE 0 END) AS failed,
            SUM(chunk_count)                             AS total_chunks,
            SUM(image_count)                             AS total_images
        FROM documents
        """
    )
    return rows[0] if rows else {}


# ── Helpers ───────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
