-- sql/003_indexes.sql
-- SQLite indexes for query performance.
-- FTS5 table (chunk_fts) is already indexed automatically — no manual index needed.
-- Vector indexes are managed internally by ChromaDB (HNSW algorithm).

-- documents
CREATE INDEX IF NOT EXISTS idx_doc_specialty ON documents(specialty);
CREATE INDEX IF NOT EXISTS idx_doc_status    ON documents(status);
CREATE INDEX IF NOT EXISTS idx_doc_type      ON documents(doc_type);
CREATE INDEX IF NOT EXISTS idx_doc_visible   ON documents(is_visible);
CREATE INDEX IF NOT EXISTS idx_doc_source    ON documents(knowledge_source_id);
CREATE INDEX IF NOT EXISTS idx_doc_hash      ON documents(content_hash);

-- processing_jobs
CREATE INDEX IF NOT EXISTS idx_job_doc    ON processing_jobs(document_id);
CREATE INDEX IF NOT EXISTS idx_job_status ON processing_jobs(status);
CREATE INDEX IF NOT EXISTS idx_job_type   ON processing_jobs(job_type);
CREATE INDEX IF NOT EXISTS idx_job_queued ON processing_jobs(queued_at);

-- conversations
CREATE INDEX IF NOT EXISTS idx_conv_token   ON conversations(session_token);
CREATE INDEX IF NOT EXISTS idx_conv_started ON conversations(started_at);

-- messages
CREATE INDEX IF NOT EXISTS idx_msg_conv    ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_msg_role    ON messages(role);
CREATE INDEX IF NOT EXISTS idx_msg_created ON messages(created_at);

-- citations
CREATE INDEX IF NOT EXISTS idx_cite_msg  ON citations(message_id);
CREATE INDEX IF NOT EXISTS idx_cite_doc  ON citations(document_id);
CREATE INDEX IF NOT EXISTS idx_cite_chk  ON citations(chunk_id);
