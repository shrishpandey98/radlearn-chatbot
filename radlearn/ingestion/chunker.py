"""
radlearn/ingestion/chunker.py
──────────────────────────────
Splits extracted document text into retrieval-optimized chunks.

Connection to system:
  - Called by ingestion/pipeline.py after text extraction (PDF, DOCX, or web).
  - Output is a list of chunk dicts ready for embedding, then ChromaDB insertion.
  - Uses LangChain's RecursiveCharacterTextSplitter which splits on paragraph
    and sentence boundaries before resorting to character splits.
  - chunk_index is 0-based and sequential within a single document.

Design decisions:
  - CHUNK_SIZE = 512 tokens — large enough for full sentences of context,
    small enough for precise retrieval.
  - CHUNK_OVERLAP = 64 tokens — prevents context loss at boundaries.
  - Chunks below MIN_CHUNK_WORDS (10 words) are marked is_header_only=True
    and excluded from vector search.
"""

from __future__ import annotations

import re
import uuid
import logging

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from radlearn.config import CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_WORDS

logger = logging.getLogger(__name__)

# Tiktoken tokenizer — cl100k_base is a close proxy for the Google embedding
# model's token count and is much faster than calling the API for sizing.
_tokenizer = tiktoken.get_encoding("cl100k_base")


def _token_count(text: str) -> int:
    """Return approximate token count for a text string."""
    return len(_tokenizer.encode(text))


# RecursiveCharacterTextSplitter splits on these separators in order:
# paragraphs → single newlines → sentences → words → characters.
_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=_token_count,
    separators=["\n\n", "\n", ". ", "! ", "? ", "; ", " ", ""],
    is_separator_regex=False,
)


# ── Public API ────────────────────────────────────────────────────

def chunk_text(
    text: str,
    document_id: str,
    doc_metadata: dict,
    page_number: int | None = None,
    section_heading: str | None = None,
    start_chunk_index: int = 0,
) -> list[dict]:
    """
    Split a block of text into overlapping chunks.

    Args:
        text:              Raw text to split (from one PDF page or DOCX section).
        document_id:       UUID of the parent document.
        doc_metadata:      Dict with document-level metadata to denormalise into chunks:
                               doc_title, doc_author, doc_year, doc_type,
                               doc_specialty, source_url, file_path, citation_format
        page_number:       PDF page number (1-based). None for DOCX and web.
        section_heading:   Detected heading for this block of text.
        start_chunk_index: 0-based starting index for chunk_index numbering.
                           Pass the total chunk count so far when chunking page by page.

    Returns:
        List of chunk dicts. Each dict contains:
            id, document_id, text, chunk_index, page_number,
            section_heading, token_count, start_char_pos, end_char_pos,
            is_header_only, and all doc_metadata fields.
        Chunks with fewer than MIN_CHUNK_WORDS words have is_header_only=True.
        The list may be empty if the input text is blank.
    """
    if not text or not text.strip():
        return []

    # Auto-detect section heading from the text if not provided
    if section_heading is None:
        section_heading = extract_section_heading(text)

    raw_chunks = _splitter.split_text(text)
    result: list[dict] = []

    for i, chunk_text_content in enumerate(raw_chunks):
        chunk_text_content = chunk_text_content.strip()
        if not chunk_text_content:
            continue

        word_count = len(chunk_text_content.split())
        token_cnt = _token_count(chunk_text_content)

        # Find character positions in the original text for tracing
        start_pos = text.find(chunk_text_content)
        end_pos = start_pos + len(chunk_text_content) if start_pos != -1 else -1

        chunk = {
            "id":               str(uuid.uuid4()),
            "document_id":      document_id,
            "text":             chunk_text_content,
            "chunk_index":      start_chunk_index + i,
            "page_number":      page_number,
            "section_heading":  section_heading,
            "token_count":      token_cnt,
            "start_char_pos":   max(start_pos, 0),
            "end_char_pos":     end_pos,
            "is_header_only":   word_count < MIN_CHUNK_WORDS,
            # Denormalised document fields (needed by ChromaDB metadata)
            "doc_title":        doc_metadata.get("doc_title", ""),
            "doc_author":       doc_metadata.get("doc_author", ""),
            "doc_year":         doc_metadata.get("doc_year", 0),
            "doc_type":         doc_metadata.get("doc_type", "article"),
            "doc_specialty":    doc_metadata.get("doc_specialty", "general"),
            "source_url":       doc_metadata.get("source_url", ""),
            "file_path":        doc_metadata.get("file_path", ""),
            "citation_format":  doc_metadata.get("citation_format", ""),
            "project_id":       doc_metadata.get("project_id", ""),
            "session_id":       doc_metadata.get("session_id", ""),
        }
        result.append(chunk)

    logger.debug(
        "chunk_text: %d chunks from %d chars (page=%s, heading=%s)",
        len(result), len(text), page_number, section_heading,
    )
    return result


def chunk_pages(
    pages: list[dict],
    document_id: str,
    doc_metadata: dict,
) -> list[dict]:
    """
    Chunk a list of page dicts produced by pdf_parser.parse_pdf().

    Args:
        pages:        List of page dicts with keys: page_num, text, heading.
        document_id:  UUID of the parent document.
        doc_metadata: Document-level metadata dict (see chunk_text docs).

    Returns:
        Flat list of all chunks across all pages.
    """
    all_chunks: list[dict] = []
    for page in pages:
        new_chunks = chunk_text(
            text=page.get("text", ""),
            document_id=document_id,
            doc_metadata=doc_metadata,
            page_number=page.get("page_num"),
            section_heading=page.get("heading"),
            start_chunk_index=len(all_chunks),
        )
        all_chunks.extend(new_chunks)
    return all_chunks


def chunk_sections(
    sections: list[dict],
    document_id: str,
    doc_metadata: dict,
) -> list[dict]:
    """
    Chunk a list of section dicts produced by docx_parser.parse_docx()
    or web_scraper.scrape_url().

    Args:
        sections:     List of section dicts with keys: heading, text.
        document_id:  UUID of the parent document.
        doc_metadata: Document-level metadata dict.

    Returns:
        Flat list of all chunks across all sections.
    """
    all_chunks: list[dict] = []
    for section in sections:
        new_chunks = chunk_text(
            text=section.get("text", ""),
            document_id=document_id,
            doc_metadata=doc_metadata,
            page_number=None,  # DOCX and web have no stable page numbers
            section_heading=section.get("heading"),
            start_chunk_index=len(all_chunks),
        )
        all_chunks.extend(new_chunks)
    return all_chunks


# ── Heading detection ─────────────────────────────────────────────

def extract_section_heading(text: str) -> str | None:
    """
    Detect a section heading from the first line of a text block.

    A heading is identified if the first non-empty line is:
        - Fewer than 80 characters long, AND
        - Does not end with a period (not a sentence), AND
        - Starts with a digit (numbered section: "3.2 Findings") OR
          is ALL CAPS OR
          is followed by two consecutive newlines.

    Returns:
        The heading string if detected, else None.
    """
    if not text:
        return None

    lines = text.strip().split("\n")
    first_line = lines[0].strip() if lines else ""

    if not first_line or len(first_line) > 80:
        return None

    # Criterion 1: ends with period → sentence, not a heading
    if first_line.endswith("."):
        return None

    # Criterion 2: numbered section (e.g. "3.2 MRI Findings")
    if re.match(r"^\d+[\.\d]*\s+\w", first_line):
        return first_line

    # Criterion 3: ALL CAPS heading
    if first_line.isupper() and len(first_line) > 3:
        return first_line

    # Criterion 4: followed by a blank line (paragraph heading)
    if len(lines) > 1 and lines[1].strip() == "" and len(first_line) > 3:
        return first_line

    return None
