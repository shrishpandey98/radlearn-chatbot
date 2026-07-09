"""
radlearn/ingestion/web_scraper.py
──────────────────────────────────
Scrapes website content using trafilatura.
Extracts title, content, and metadata.
"""
import logging
import trafilatura
from typing import List, Dict

logger = logging.getLogger(__name__)

def scrape_url(url: str) -> List[Dict]:
    """
    Downloads and extracts text from a URL.
    Returns a single-item list simulating a section for the chunker.
    """
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        logger.error(f"Failed to download URL: {url}")
        return []
        
    metadata = trafilatura.extract_metadata(downloaded)
    content = trafilatura.extract(downloaded, include_comments=False, include_tables=True)
    
    if not content:
        logger.error(f"Failed to extract content from URL: {url}")
        return []
        
    title = metadata.title if metadata and metadata.title else "Web Page"
    
    # We return a single section representing the entire page. 
    # RecursiveCharacterTextSplitter will handle chunking it effectively.
    return [{
        "heading": title,
        "text": content,
        "source_url": url,
        "date": metadata.date if metadata else None
    }]
