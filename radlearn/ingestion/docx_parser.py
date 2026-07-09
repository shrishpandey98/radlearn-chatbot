"""
radlearn/ingestion/docx_parser.py
──────────────────────────────────
Parses DOCX documents using python-docx.
Extracts headings and paragraphs while preserving section structure.
"""
import logging
import docx
from typing import List, Dict

logger = logging.getLogger(__name__)

def parse_docx(file_path: str) -> List[Dict]:
    """
    Parses a DOCX file and returns a list of section dictionaries.
    A new section begins whenever a Heading style is encountered.
    """
    sections = []
    
    try:
        doc = docx.Document(file_path)
    except Exception as e:
        logger.error(f"Failed to open DOCX {file_path}: {e}")
        return []

    current_heading = None
    current_text = []
    
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
            
        # Detect headings based on style name
        if para.style.name.startswith('Heading'):
            # Save previous section if it has text
            if current_text:
                sections.append({
                    "heading": current_heading,
                    "text": "\n\n".join(current_text)
                })
                current_text = []
            current_heading = text
        else:
            current_text.append(text)
            
    # Add final section
    if current_text:
        sections.append({
            "heading": current_heading,
            "text": "\n\n".join(current_text)
        })
        
    return sections
