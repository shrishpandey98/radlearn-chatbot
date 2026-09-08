"""
radlearn/retrieval/web_search.py
────────────────────────────────
Fallback web search using trusted radiology domains and PubMed E-Utilities.
"""
import logging
import json
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from typing import List, Dict

logger = logging.getLogger(__name__)

def _chunk_text(text: str, max_chars: int = 800) -> List[str]:
    chunks = []
    paragraphs = text.split("\n\n")
    current = ""
    for p in paragraphs:
        if len(current) + len(p) < max_chars:
            current += p + " "
        else:
            if current:
                chunks.append(current.strip())
            current = p + " "
    if current:
        chunks.append(current.strip())
    return chunks

def perform_web_search(query: str, top_k: int = 4) -> List[Dict]:
    """
    Search PubMed via NCBI E-Utilities using Python standard library,
    extract abstracts, and return formatted chunks.
    """
    web_chunks = []
    if not query or not query.strip():
        return web_chunks

    try:
        # 1. Search PubMed for matching IDs
        clean_query = query.strip()
        search_url = (
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?"
            f"db=pubmed&term={urllib.parse.quote_plus(clean_query)}&retmode=json&retmax={top_k}"
        )
        req = urllib.request.Request(
            search_url,
            headers={"User-Agent": "RadLearn-RAG/1.0 (Radiology Educational Assistant)"}
        )
        
        with urllib.request.urlopen(req, timeout=6) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="ignore"))
            ids = data.get("esearchresult", {}).get("idlist", [])

        # If strict search returned nothing, retry with core medical terms
        if not ids:
            simplified = " ".join([w for w in clean_query.split() if len(w) > 2 and w.lower() not in ["what", "does", "indicate", "for", "the", "in", "and", "are", "with", "from"]])
            if simplified != clean_query and len(simplified) > 3:
                search_url = (
                    f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?"
                    f"db=pubmed&term={urllib.parse.quote_plus(simplified)}&retmode=json&retmax={top_k}"
                )
                req = urllib.request.Request(
                    search_url,
                    headers={"User-Agent": "RadLearn-RAG/1.0 (Radiology Educational Assistant)"}
                )
                with urllib.request.urlopen(req, timeout=6) as resp:
                    data = json.loads(resp.read().decode("utf-8", errors="ignore"))
                    ids = data.get("esearchresult", {}).get("idlist", [])

        # 2. Fetch abstracts for found IDs
        if ids:
            fetch_url = (
                f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"
                f"db=pubmed&id={','.join(ids)}&retmode=xml"
            )
            req_fetch = urllib.request.Request(
                fetch_url,
                headers={"User-Agent": "RadLearn-RAG/1.0 (Radiology Educational Assistant)"}
            )
            with urllib.request.urlopen(req_fetch, timeout=8) as fetch_resp:
                xml_content = fetch_resp.read().decode("utf-8", errors="ignore")
                root = ET.fromstring(xml_content)
                
                for i, article in enumerate(root.findall(".//PubmedArticle")):
                    pmid = article.findtext(".//PMID") or f"pmid_{i}"
                    title = article.findtext(".//ArticleTitle") or f"PubMed Article {pmid}"
                    abstract_nodes = article.findall(".//AbstractText")
                    
                    abstract_parts = [node.text for node in abstract_nodes if node.text]
                    if not abstract_parts:
                        continue
                        
                    abstract = " ".join(abstract_parts)
                    url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                    
                    # Chunk abstract for RAG
                    text_chunks = _chunk_text(abstract)
                    for j, ctext in enumerate(text_chunks[:3]):
                        web_chunks.append({
                            "id": f"pubmed_{pmid}_{j}",
                            "document_id": f"pubmed_{pmid}",
                            "text": ctext,
                            "chunk_index": j,
                            "doc_title": f"[PubMed] {title}",
                            "source_url": url,
                            "formatted_citation": f"PubMed: {title[:70]}...",
                            "citation_format": f"[W{i+1}]",
                            "similarity_score": 0.95,
                            "rrf_score": 0.90 - (i * 0.05)
                        })

    except Exception as e:
        logger.warning(f"PubMed web search error: {e}")

    return web_chunks

