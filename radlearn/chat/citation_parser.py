"""
radlearn/chat/citation_parser.py
─────────────────────────────────
Validates and extracts citations from the LLM answer.
"""
import re
from typing import List, Dict

def parse_citations(answer_text: str, retrieved_chunks: List[Dict]) -> List[Dict]:
    """
    Finds [N] markers in text, validates against retrieved_chunks, deduplicates, 
    and builds citation objects.
    """
    # Find all bracketed numbers e.g. [1], [2]
    matches = re.findall(r'\[(\d+)\]', answer_text)
    
    validated_citations = []
    seen_numbers = set()
    
    max_citation_num = len(retrieved_chunks)
    
    for match in matches:
        cit_num = int(match)
        # Deduplicate and validate range
        if cit_num not in seen_numbers and 1 <= cit_num <= max_citation_num:
            seen_numbers.add(cit_num)
            chunk = retrieved_chunks[cit_num - 1] # 0-based index
            
            title = chunk.get("doc_title") or "Unknown Document"
            formatted = title
            
            file_path = chunk.get("file_path", "")
            source_url = chunk.get("source_url", "")
            if not chunk.get("document_id", "").startswith("pubmed_") and (not file_path or not source_url):
                try:
                    import sqlite3
                    from radlearn.config import SQLITE_PATH
                    conn = sqlite3.connect(SQLITE_PATH)
                    c = conn.cursor()
                    c.execute("SELECT file_path, source_url FROM documents WHERE id = ?", (chunk["document_id"],))
                    row = c.fetchone()
                    if row:
                        if not file_path and row[0]:
                            file_path = row[0]
                        if not source_url and row[1] and row[1] != "session_upload":
                            source_url = row[1]
                    conn.close()
                except Exception:
                    pass


            validated_citations.append({
                "citation_number": cit_num,
                "chunk_id": chunk["id"],
                "document_id": chunk["document_id"],
                "cited_text": chunk["text"],
                "relevance_score": chunk.get("rrf_score", 0.0),
                "rank_position": cit_num,
                "formatted_citation": formatted,
                "source_url": source_url,
                "page_number": chunk.get("page_number", -1),
                "file_path": file_path
            })
            
    return validated_citations
