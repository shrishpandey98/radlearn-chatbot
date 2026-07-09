"""
radlearn/retrieval/web_search.py
────────────────────────────────
Fallback web search using trusted radiology domains.
"""
import logging
from typing import List, Dict
import trafilatura
from duckduckgo_search import DDGS
from radlearn.config import EMBEDDING_MAX_TOKENS

# Heuristic to split long web pages
def _chunk_text(text: str, max_chars: int = 800) -> List[str]:
    chunks = []
    # simple split by double newline
    paragraphs = text.split("\\n\\n")
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

def perform_web_search(query: str, top_k: int = 3) -> List[Dict]:
    """
    Search PubMed via E-Utilities, extract abstracts, and return formatted chunks.
    This replaces DuckDuckGo as the primary trusted web source due to aggressive bot-blocking on cloud IPs.
    """
    web_chunks = []
    
    try:
        import requests
        import urllib.parse
        from bs4 import BeautifulSoup
        
        # 1. Search for article IDs
        search_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={urllib.parse.quote_plus(query)}&retmode=json&retmax={top_k}"
        resp = requests.get(search_url, timeout=5)
        
        if resp.status_code == 200:
            data = resp.json()
            ids = data.get("esearchresult", {}).get("idlist", [])
            
            # 2. Fetch article abstracts
            if ids:
                fetch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={','.join(ids)}&retmode=xml"
                fetch_resp = requests.get(fetch_url, timeout=5)
                
                if fetch_resp.status_code == 200:
                    soup = BeautifulSoup(fetch_resp.content, "xml")
                    for i, article in enumerate(soup.find_all("PubmedArticle")):
                        pmid = article.find("PMID").text if article.find("PMID") else f"unknown_{i}"
                        title_elem = article.find("ArticleTitle")
                        title = title_elem.text if title_elem else "PubMed Article"
                        
                        abstract_texts = article.find_all("AbstractText")
                        if not abstract_texts:
                            continue
                            
                        # Join multiple abstract sections if they exist
                        abstract = " ".join([elem.text for elem in abstract_texts])
                        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                        
                        # Chunk the abstract
                        text_chunks = _chunk_text(abstract)
                        for j, ctext in enumerate(text_chunks[:5]):
                            web_chunks.append({
                                "id": f"pubmed_{pmid}_{j}",
                                "document_id": f"pubmed_doc_{pmid}",
                                "text": ctext,
                                "chunk_index": j,
                                "doc_title": title,
                                "source_url": url,
                                "citation_format": f"[W{i+1}]", # Web citation format
                                "similarity_score": 0.0 # Will be re-scored by RRF
                            })
                            
    except Exception as e:
        logging.warning(f"PubMed search fallback failed: {e}")
        
    return web_chunks
