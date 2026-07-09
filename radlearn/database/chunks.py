"""
radlearn/database/chunks.py
────────────────────────────
All chunk operations: insert into ChromaDB, mirror text to SQLite FTS5,
semantic search, keyword search, and deletion.

Connection to system:
  - Called by ingestion/pipeline.py (add_chunks_to_chromadb) after embedding.
  - Called by retrieval/searcher.py (semantic_search, keyword_search).
  - ChromaDB stores: embedding vector + chunk text + metadata dict.
  - SQLite FTS5 stores: chunk text only (for keyword retrieval ranking).

Important: ChromaDB metadata values must be str, int, float, or bool.
           None values cause silent errors — use "" and -1 as sentinels.
"""

from __future__ import annotations

from radlearn.database.client import get_chunks_collection, execute_sql
from radlearn.config import SEMANTIC_TOP_K, KEYWORD_TOP_K, EMBEDDING_BATCH_SIZE


# ── Insert ────────────────────────────────────────────────────────

def add_chunks_to_chromadb(chunks: list[dict]) -> None:
    """
    Batch-insert embedded chunks into ChromaDB and mirror text to FTS5.

    Each chunk dict must contain:
        id              UUID string (ChromaDB document ID)
        document_id     UUID string
        text            chunk text string
        embedding       list[float] of length 768
        chunk_index     int
        doc_title       str
        doc_author      str or None
        doc_year        int or None
        doc_type        str
        doc_specialty   str
        source_url      str or None
        file_path       str or None
        citation_format str or None
        page_number     int or None   (-1 if absent)
        section_heading str or None   ("" if absent)
        token_count     int
        is_header_only  bool

    Inserts in batches of EMBEDDING_BATCH_SIZE to stay within ChromaDB limits.
    """
    if not chunks:
        return

    collection = get_chunks_collection()

    # Build ChromaDB payload lists
    ids = [c["id"] for c in chunks]
    embeddings = [c["embedding"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [_build_chunk_metadata(c) for c in chunks]

    # Batch insert into ChromaDB
    batch_size = EMBEDDING_BATCH_SIZE
    for i in range(0, len(chunks), batch_size):
        collection.add(
            ids=ids[i : i + batch_size],
            embeddings=embeddings[i : i + batch_size],
            documents=documents[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
        )

    # Mirror text to SQLite FTS5 for keyword search
    for c in chunks:
        if not c.get("is_header_only", False):
            execute_sql(
                """
                INSERT INTO chunk_fts(chunk_id, document_id, text, specialty)
                VALUES (:chunk_id, :document_id, :text, :specialty)
                """,
                {
                    "chunk_id":   c["id"],
                    "document_id": c["document_id"],
                    "text":       c["text"],
                    "specialty":  c.get("doc_specialty", "general"),
                },
            )


# ── Search ────────────────────────────────────────────────────────

def semantic_search(
    query_vector: list[float],
    specialty_filter: str | None = None,
    top_k: int = SEMANTIC_TOP_K,
    session_id: str | None = None,
    project_id: str | None = None
) -> list[dict]:
    """
    Vector similarity search via ChromaDB.

    Args:
        query_vector:     768-dim embedding of the user query.
        specialty_filter: If set, filters to this specialty + 'general'.
        top_k:            Number of results to return.
        session_id:       Optional session ID to filter private user uploads.
        project_id:       Optional Project ID to filter and prioritize project uploads.

    Returns:
        List of chunk dicts sorted by similarity descending.
        Each dict has: id, document_id, text, similarity_score, and all metadata fields.
    """
    collection = get_chunks_collection()
    total = collection.count()
    if total == 0:
        return []

    # Build metadata filter
    filters = [{"is_header_only": False}]
    if specialty_filter and specialty_filter not in ("", "general", "All"):
        filters.append({"doc_specialty": {"$in": [specialty_filter, "general"]}})

    where_filter = {"$and": filters} if len(filters) > 1 else filters[0]

    # Fetch more results to allow for post-filtering
    fetch_k = top_k * 3 if (session_id or project_id) else top_k

    try:
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=min(fetch_k, total),
            where=where_filter,
            include=["metadatas", "documents", "distances"],
        )
    except Exception:
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=min(fetch_k, total),
            include=["metadatas", "documents", "distances"],
        )

    parsed = _parse_chromadb_results(results)
    
    # Optional: We could boost similarity scores for project_id here.
    if project_id:
        for p in parsed:
            if p.get("project_id") == project_id:
                p["similarity_score"] += 0.1  # Prioritize project files
                
    # Sort again after potential boosting
    parsed = sorted(parsed, key=lambda x: x["similarity_score"], reverse=True)
    
    filtered = []
    for p in parsed:
        stype = p.get("source_type")
        sid = p.get("session_id")
        pid = p.get("project_id")
        
        # If it's a manual upload, only allow it if it belongs to the current session or project
        if stype == "user_upload":
            if (project_id and pid == project_id) or (session_id and sid == session_id):
                pass # allow
            else:
                continue
            
        filtered.append(p)
        if len(filtered) == top_k:
            break
            
    return filtered


def keyword_search(
    query_text: str,
    specialty_filter: str | None = None,
    top_k: int = KEYWORD_TOP_K,
    session_id: str | None = None,
    project_id: str | None = None
) -> list[dict]:
    """
    Full-text keyword search via SQLite FTS5.

    Strategy:
        1. Query chunk_fts (FTS5) to get ranked chunk IDs.
        2. Fetch full chunk data from ChromaDB by those IDs.
        3. Return merged list preserving FTS rank order.

    Args:
        query_text:       Raw user query string (not embedded).
        specialty_filter: Optional specialty to restrict search.
        top_k:            Maximum results to return.

    Returns:
        List of chunk dicts (same schema as semantic_search output).
        similarity_score is set to a rank-based float (not cosine).
    """
    import re as _re
    # FTS5 MATCH interprets ?, !, :, ", *, ^, (, ) as syntax tokens.
    # Strip them and collapse extra whitespace so natural-language questions work.
    fts_query = _re.sub(r'[^\w\s]', ' ', query_text)
    fts_query = ' '.join(fts_query.split())   # collapse whitespace
    if not fts_query.strip():
        return []

    # Build an FTS5 query that searches individual words (OR semantics by default).
    # Wrap each token so multi-word terms are matched individually, which is more
    # reliable than phrase search for long expanded medical queries.
    words = fts_query.split()
    # Limit to first 10 significant words to avoid overly narrow FTS queries
    fts_query = ' OR '.join(f'"{w}"' for w in words[:10] if len(w) > 2)
    if not fts_query:
        return []

    specialty_clause = ""
    params: dict = {"q": fts_query, "top_k": top_k}

    if specialty_filter and specialty_filter not in ("", "All"):
        specialty_clause = "AND specialty IN (:spec, 'general')"
        params["spec"] = specialty_filter

    session_clause = ""
    if project_id and session_id:
        session_clause = "AND (d.source_url != 'session_upload' OR d.project_id = :project_id OR d.knowledge_source_id = :session_id)"
        params["project_id"] = project_id
        params["session_id"] = session_id
    elif project_id:
        session_clause = "AND (d.source_url != 'session_upload' OR d.project_id = :project_id)"
        params["project_id"] = project_id
    elif session_id:
        session_clause = "AND (d.source_url != 'session_upload' OR d.knowledge_source_id = :session_id)"
        params["session_id"] = session_id
    else:
        session_clause = "AND d.source_url != 'session_upload'"

    # SQLite FTS5: bm25() returns negative scores (more negative = better match)
    # We ORDER ASC so the most relevant results come first
    try:
        fts_rows = execute_sql(
            f"""
            SELECT c.chunk_id, bm25(c.chunk_fts) AS score
            FROM chunk_fts c
            JOIN documents d ON c.document_id = d.id
            WHERE c.chunk_fts MATCH :q
            {specialty_clause}
            {session_clause}
            ORDER BY score
            LIMIT :top_k
            """,
            params,
        )
    except Exception:
        fts_rows = []

    if not fts_rows:
        return []

    chunk_ids = [row["chunk_id"] for row in fts_rows]
    collection = get_chunks_collection()

    try:
        chroma_results = collection.get(
            ids=chunk_ids,
            include=["metadatas", "documents"],
        )
    except Exception:
        return []

    # Build lookup for ChromaDB data
    id_to_meta: dict[str, dict] = {}
    id_to_text: dict[str, str] = {}
    for i, cid in enumerate(chroma_results["ids"]):
        id_to_meta[cid] = chroma_results["metadatas"][i]
        id_to_text[cid] = chroma_results["documents"][i]

    # Reconstruct in FTS rank order
    result = []
    for rank, row in enumerate(fts_rows):
        cid = row["chunk_id"]
        if cid not in id_to_meta:
            continue
        meta = id_to_meta[cid]
        result.append({
            **_meta_to_chunk_dict(cid, meta, id_to_text[cid]),
            "similarity_score": 0.0,
        })

    return result


# ── Delete ────────────────────────────────────────────────────────

def delete_chunks_for_document(document_id: str) -> int:
    """
    Delete all ChromaDB chunk entries and FTS5 rows for a document.

    Returns:
        Number of chunks deleted from ChromaDB.
    """
    collection = get_chunks_collection()

    # Get IDs of all chunks for this document
    try:
        existing = collection.get(
            where={"document_id": document_id},
            include=[],
        )
        chunk_ids = existing["ids"]
    except Exception:
        chunk_ids = []

    if chunk_ids:
        collection.delete(ids=chunk_ids)

    # Clean FTS5 mirror
    execute_sql(
        "DELETE FROM chunk_fts WHERE document_id = :did",
        {"did": document_id},
    )

    return len(chunk_ids)


def count_chunks() -> int:
    """Return total number of chunks in ChromaDB."""
    return get_chunks_collection().count()


# ── Private helpers ───────────────────────────────────────────────

def _build_chunk_metadata(c: dict) -> dict:
    """
    Convert a chunk dict to a ChromaDB-safe metadata dict.
    ChromaDB requires: no None values; all values must be str/int/float/bool.
    """
    return {
        "document_id":      c["document_id"],
        "doc_title":        c.get("doc_title") or "",
        "doc_author":       c.get("doc_author") or "",
        "doc_year":         int(c.get("doc_year") or 0),
        "doc_type":         c.get("doc_type") or "article",
        "doc_specialty":    c.get("doc_specialty") or "general",
        "source_url":       c.get("source_url") or "",
        "file_path":        c.get("file_path") or "",
        "citation_format":  c.get("citation_format") or "",
        "chunk_index":      int(c.get("chunk_index") or 0),
        "page_number":      int(c.get("page_number") or -1),
        "section_heading":  c.get("section_heading") or "",
        "token_count":      int(c.get("token_count") or 0),
        "is_header_only":   bool(c.get("is_header_only", False)),
        "project_id":       c.get("project_id") or "",
        "session_id":       c.get("session_id") or "",
    }


def _parse_chromadb_results(results: dict) -> list[dict]:
    """Convert raw ChromaDB query() output to list of chunk dicts."""
    chunks = []
    for i, chunk_id in enumerate(results["ids"][0]):
        meta = results["metadatas"][0][i]
        text = results["documents"][0][i]
        distance = results["distances"][0][i]
        # cosine space: distance = 1 - similarity → similarity = 1 - distance
        similarity = round(1.0 - distance, 4)
        chunks.append(_meta_to_chunk_dict(chunk_id, meta, text, similarity))
    return chunks


def _meta_to_chunk_dict(
    chunk_id: str,
    meta: dict,
    text: str,
    similarity: float = 0.0,
) -> dict:
    """Normalise a ChromaDB metadata dict into a consistent chunk dict."""
    return {
        "id":               chunk_id,
        "document_id":      meta["document_id"],
        "text":             text,
        "chunk_index":      meta["chunk_index"],
        "page_number":      meta["page_number"] if meta["page_number"] != -1 else None,
        "section_heading":  meta["section_heading"] or None,
        "token_count":      meta["token_count"],
        "doc_title":        meta["doc_title"],
        "doc_author":       meta["doc_author"] or None,
        "doc_year":         meta["doc_year"] or None,
        "doc_specialty":    meta["doc_specialty"],
        "project_id":       meta.get("project_id") or None,
        "session_id":       meta.get("session_id") or None,
        "doc_type":         meta["doc_type"],
        "source_url":       meta["source_url"] or None,
        "file_path":        meta["file_path"] or None,
        "citation_format":  meta["citation_format"] or None,
        "similarity_score": similarity,
    }
