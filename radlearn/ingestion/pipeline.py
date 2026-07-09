"""
radlearn/ingestion/pipeline.py
───────────────────────────────
Orchestrates the entire ingestion process for PDFs, DOCX, and Web.
Handles hashing, duplicate detection, DB tracking, chunking, and embedding.
"""
import hashlib
import logging
from pathlib import Path

from radlearn.database.documents import create_document, check_duplicate_hash, update_document, delete_document_record
from radlearn.database.chunks import add_chunks_to_chromadb
from radlearn.database.client import execute_sql
from radlearn.ingestion.pdf_parser import parse_pdf
from radlearn.ingestion.docx_parser import parse_docx
from radlearn.ingestion.web_scraper import scrape_url
from radlearn.ingestion.chunker import chunk_pages, chunk_sections
from radlearn.ingestion.embedder import embed_texts
from radlearn.ingestion.image_extractor import extract_images_from_pdf
from radlearn.config import DOCUMENTS_DIR

logger = logging.getLogger(__name__)

def compute_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()

def _store_document_file(file_bytes: bytes, doc_id: str, filename: str) -> str:
    """Save uploaded file to local data directory. Returns relative path."""
    doc_dir = DOCUMENTS_DIR / doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(filename).name
    file_path = doc_dir / safe_name
    file_path.write_bytes(file_bytes)
    return str(file_path)

def ingest_file(file_bytes: bytes, filename: str, doc_metadata: dict) -> dict:
    """
    Main ingestion pipeline for a file (PDF or DOCX).
    doc_metadata must contain: title, doc_type, specialty, license_type
    """
    content_hash = compute_hash(file_bytes)

    # Duplicate check
    existing = check_duplicate_hash(content_hash)
    if existing:
        if existing["status"] == "completed":
            logger.info(f"Duplicate (completed) detected. Skipping {filename}.")
            return {"status": "skipped", "reason": "duplicate", "doc_id": existing["id"]}
        else:
            # Previously failed — delete the stale record and retry cleanly
            logger.warning(f"Re-ingesting previously failed document: {filename}.")
            delete_document_record(existing["id"])

    doc_metadata["content_hash"] = content_hash
    doc_record = create_document(doc_metadata)
    doc_id = doc_record["id"]
    
    try:
        # Save file to disk
        local_path = _store_document_file(file_bytes, doc_id, filename)
        update_document(doc_id, {"file_path": local_path, "status": "processing"})

        # Inject denormalised fields chunker.py reads from doc_metadata
        # (chunker uses 'doc_title', but the caller API uses 'title')
        doc_metadata.setdefault("doc_title",    doc_metadata.get("title", filename))
        doc_metadata.setdefault("doc_author",   doc_metadata.get("author", ""))
        doc_metadata.setdefault("doc_year",     doc_metadata.get("publication_year", 0))
        doc_metadata.setdefault("doc_type",     doc_metadata.get("doc_type", "article"))
        doc_metadata.setdefault("doc_specialty",doc_metadata.get("specialty", "general"))

        # Parse & Chunk
        if filename.lower().endswith('.pdf'):
            parsed_data = parse_pdf(local_path)
            chunks = chunk_pages(parsed_data, doc_id, doc_metadata)
            images_extracted = extract_images_from_pdf(local_path, doc_id, doc_metadata)
        elif filename.lower().endswith('.docx'):
            parsed_data = parse_docx(local_path)
            chunks = chunk_sections(parsed_data, doc_id, doc_metadata)
            images_extracted = 0
        elif filename.lower().endswith('.txt'):
            text_content = file_bytes.decode('utf-8', errors='replace')
            parsed_data = [{"heading": filename, "text": text_content}]
            chunks = chunk_sections(parsed_data, doc_id, doc_metadata)
            images_extracted = 0
        else:
            raise ValueError(f"Unsupported file format: {filename}")

        if not chunks:
            raise ValueError("No text extracted from document")

        # Embed
        texts = [c["text"] for c in chunks]
        embeddings = embed_texts(texts)
        
        for chunk, emb in zip(chunks, embeddings):
            chunk["embedding"] = emb

        # Store to ChromaDB & SQLite
        add_chunks_to_chromadb(chunks)
        
        # Mark Complete
        update_document(doc_id, {
            "status": "completed",
            "chunk_count": len(chunks),
            "image_count": images_extracted,
            "total_pages": len(parsed_data) if isinstance(parsed_data, list) else 0
        })
        
        return {"status": "success", "doc_id": doc_id, "chunks": len(chunks), "images": images_extracted}

    except Exception as e:
        logger.error(f"Ingestion failed for {filename}: {e}")
        update_document(doc_id, {"status": "failed"})
        return {"status": "failed", "reason": str(e), "doc_id": doc_id}


def ingest_url(url: str, doc_metadata: dict) -> dict:
    """Ingestion pipeline for a website."""
    content_hash = compute_hash(url.encode('utf-8'))
    existing = check_duplicate_hash(content_hash)
    if existing:
        if existing["status"] == "completed":
            return {"status": "skipped", "reason": "duplicate url", "doc_id": existing["id"]}
        else:
            delete_document_record(existing["id"])

    doc_metadata["content_hash"] = content_hash
    doc_metadata["source_url"] = url
    doc_record = create_document(doc_metadata)
    doc_id = doc_record["id"]
    
    try:
        update_document(doc_id, {"status": "processing"})
        parsed_data = scrape_url(url)
        chunks = chunk_sections(parsed_data, doc_id, doc_metadata)
        
        if not chunks:
            raise ValueError("No text extracted from URL")

        texts = [c["text"] for c in chunks]
        embeddings = embed_texts(texts)
        
        for chunk, emb in zip(chunks, embeddings):
            chunk["embedding"] = emb

        add_chunks_to_chromadb(chunks)
        
        update_document(doc_id, {
            "status": "completed",
            "chunk_count": len(chunks),
            "image_count": 0
        })
        
        return {"status": "success", "doc_id": doc_id, "chunks": len(chunks)}

    except Exception as e:
        logger.error(f"Ingestion failed for {url}: {e}")
        update_document(doc_id, {"status": "failed"})
        return {"status": "failed", "reason": str(e), "doc_id": doc_id}
