-- sql/002_tables.sql
-- RadLearn SQLite Schema
-- Executed by: python scripts/setup_database.py
--
-- What lives here vs ChromaDB:
--   SQLite  → relational metadata (documents, jobs, conversations, messages, citations)
--   ChromaDB → vector embeddings (chunks, image captions)
--   Filesystem → actual files (PDFs, images)
--
-- NOTE: SQLite uses INTEGER 1/0 for booleans, TEXT for UUIDs and timestamps.

-- ================================================================
-- TABLE: knowledge_sources
-- Registry of content providers (e.g. "ACR Guidelines", "Manual Uploads")
-- ================================================================
CREATE TABLE IF NOT EXISTS knowledge_sources (
    id           TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    source_type  TEXT NOT NULL DEFAULT 'manual_upload'
                 CHECK(source_type IN ('pdf_collection','website','manual_upload')),
    base_url     TEXT NOT NULL DEFAULT '',
    description  TEXT NOT NULL DEFAULT '',
    license_type TEXT NOT NULL DEFAULT 'unknown'
                 CHECK(license_type IN ('cc_by','cc_by_nc','open_access','fair_use','proprietary','unknown')),
    specialty_tags TEXT NOT NULL DEFAULT '[]',
    is_active    INTEGER NOT NULL DEFAULT 1,
    created_at   TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

-- ================================================================
-- TABLE: documents
-- Master record for every ingested PDF, DOCX, or web page.
-- Chunks and images reference this table.
-- ================================================================
CREATE TABLE IF NOT EXISTS documents (
    id                   TEXT PRIMARY KEY,
    knowledge_source_id  TEXT REFERENCES knowledge_sources(id) ON DELETE SET NULL,
    title                TEXT NOT NULL,
    author               TEXT NOT NULL DEFAULT '',
    publication_year     INTEGER NOT NULL DEFAULT 0,
    source_url           TEXT NOT NULL DEFAULT '',
    file_path            TEXT NOT NULL DEFAULT '',
    doc_type             TEXT NOT NULL DEFAULT 'article'
                         CHECK(doc_type IN ('guideline','textbook','article','case_report','website','review','atlas')),
    specialty            TEXT NOT NULL DEFAULT 'general'
                         CHECK(specialty IN ('general','neuro','breast','chest','msk','abdominal',
                                            'cardiothoracic','pediatric','interventional','nuclear')),
    subspecialty_tags    TEXT NOT NULL DEFAULT '[]',
    license_type         TEXT NOT NULL DEFAULT 'unknown',
    citation_format      TEXT NOT NULL DEFAULT '',
    status               TEXT NOT NULL DEFAULT 'pending'
                         CHECK(status IN ('pending','processing','completed','failed','skipped','hidden')),
    content_hash         TEXT UNIQUE DEFAULT '',
    is_visible           INTEGER NOT NULL DEFAULT 1,
    total_pages          INTEGER NOT NULL DEFAULT 0,
    total_words          INTEGER NOT NULL DEFAULT 0,
    chunk_count          INTEGER NOT NULL DEFAULT 0,
    image_count          INTEGER NOT NULL DEFAULT 0,
    ingested_at          TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at           TEXT NOT NULL DEFAULT (datetime('now'))
);

-- ================================================================
-- TABLE: chunk_fts
-- Full-text search mirror of chunk text for keyword retrieval.
-- Vector embeddings for the same chunks live in ChromaDB.
-- The chunk_id column maps directly to ChromaDB document IDs.
-- ================================================================
CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
    chunk_id    UNINDEXED,
    document_id UNINDEXED,
    text,
    specialty   UNINDEXED,
    tokenize    = "porter ascii"
);

-- ================================================================
-- TABLE: processing_jobs
-- State machine for every ingestion operation.
-- Tracks progress, retries, and error messages for the admin UI.
-- ================================================================
CREATE TABLE IF NOT EXISTS processing_jobs (
    id                 TEXT PRIMARY KEY,
    document_id        TEXT REFERENCES documents(id) ON DELETE CASCADE,
    job_type           TEXT NOT NULL
                       CHECK(job_type IN ('pdf_ingest','web_scrape','docx_ingest',
                                         'extract_images','reprocess','delete_doc')),
    triggered_by       TEXT NOT NULL DEFAULT 'manual'
                       CHECK(triggered_by IN ('manual','upload','admin')),
    status             TEXT NOT NULL DEFAULT 'queued'
                       CHECK(status IN ('queued','running','completed','failed','cancelled','retrying')),
    attempt_number     INTEGER NOT NULL DEFAULT 1,
    max_attempts       INTEGER NOT NULL DEFAULT 3,
    total_steps        INTEGER NOT NULL DEFAULT 0,
    completed_steps    INTEGER NOT NULL DEFAULT 0,
    chunks_created     INTEGER NOT NULL DEFAULT 0,
    images_extracted   INTEGER NOT NULL DEFAULT 0,
    processing_time_ms INTEGER NOT NULL DEFAULT 0,
    error_message      TEXT NOT NULL DEFAULT '',
    queued_at          TEXT NOT NULL DEFAULT (datetime('now')),
    started_at         TEXT NOT NULL DEFAULT '',
    completed_at       TEXT NOT NULL DEFAULT '',
    updated_at         TEXT NOT NULL DEFAULT (datetime('now'))
);

-- ================================================================
-- TABLE: conversations
-- Groups individual messages into a session.
-- user_id is NULL in MVP (no login required).
-- ================================================================
CREATE TABLE IF NOT EXISTS conversations (
    id               TEXT PRIMARY KEY,
    user_id          TEXT NOT NULL DEFAULT '',
    session_token    TEXT NOT NULL DEFAULT '',
    specialty_filter TEXT NOT NULL DEFAULT '',
    message_count    INTEGER NOT NULL DEFAULT 0,
    started_at       TEXT NOT NULL DEFAULT (datetime('now')),
    last_active_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

-- ================================================================
-- TABLE: messages
-- Every user question and AI answer.
-- Metrics columns populated for assistant messages.
-- ================================================================
CREATE TABLE IF NOT EXISTS messages (
    id                   TEXT PRIMARY KEY,
    conversation_id      TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role                 TEXT NOT NULL CHECK(role IN ('user','assistant')),
    content              TEXT NOT NULL,
    expanded_query       TEXT NOT NULL DEFAULT '',
    top_similarity_score REAL NOT NULL DEFAULT 0.0,
    chunks_retrieved     INTEGER NOT NULL DEFAULT 0,
    images_retrieved     INTEGER NOT NULL DEFAULT 0,
    retrieval_time_ms    INTEGER NOT NULL DEFAULT 0,
    llm_time_ms          INTEGER NOT NULL DEFAULT 0,
    total_time_ms        INTEGER NOT NULL DEFAULT 0,
    was_answered         INTEGER NOT NULL DEFAULT 0,
    llm_model            TEXT NOT NULL DEFAULT 'gemini-1.5-flash',
    user_rating          INTEGER NOT NULL DEFAULT 0,
    was_helpful          INTEGER NOT NULL DEFAULT -1,
    feedback_text        TEXT NOT NULL DEFAULT '',
    message_index        INTEGER NOT NULL DEFAULT 0,
    created_at           TEXT NOT NULL DEFAULT (datetime('now'))
);

-- ================================================================
-- TABLE: citations
-- Immutable audit trail: every [N] marker → specific chunk + document.
-- chunk_id is a UUID string that matches a ChromaDB document ID.
-- ================================================================
CREATE TABLE IF NOT EXISTS citations (
    id                 TEXT PRIMARY KEY,
    message_id         TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    chunk_id           TEXT NOT NULL,
    document_id        TEXT NOT NULL REFERENCES documents(id) ON DELETE RESTRICT,
    citation_number    INTEGER NOT NULL,
    cited_text         TEXT NOT NULL DEFAULT '',
    relevance_score    REAL NOT NULL DEFAULT 0.0,
    rank_position      INTEGER NOT NULL DEFAULT 0,
    formatted_citation TEXT NOT NULL DEFAULT '',
    source_url         TEXT NOT NULL DEFAULT '',
    page_number        INTEGER NOT NULL DEFAULT -1,
    created_at         TEXT NOT NULL DEFAULT (datetime('now'))
);
