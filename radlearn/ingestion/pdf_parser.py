"""
radlearn/ingestion/pdf_parser.py
─────────────────────────────────
Parses PDF documents using PyMuPDF (fitz) with block-level extraction
to preserve multi-column reading order and extract headings.
"""
import logging
import fitz  # PyMuPDF
from typing import List, Dict

logger = logging.getLogger(__name__)

def parse_pdf(file_path: str) -> List[Dict]:
    """
    Parses a PDF file and returns a list of dictionaries representing each page.
    Handles multi-column layouts by sorting text blocks spatially.
    """
    pages_data = []
    
    try:
        doc = fitz.open(file_path)
    except Exception as e:
        logger.error(f"Failed to open PDF {file_path}: {e}")
        return []

    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # Get blocks: (x0, y0, x1, y1, "lines in block", block_no, block_type)
        blocks = page.get_text("blocks")
        
        # Filter for text blocks (type == 0)
        text_blocks = [b for b in blocks if b[6] == 0]
        
        if not text_blocks:
            continue
            
        page_width = page.rect.width
        
        # Multi-column aware sorting:
        # Group blocks into visual columns. For standard 2-column radiology journals,
        # we divide the page into a left and right half.
        def block_sort_key(b):
            x0, y0 = b[0], b[1]
            # Column index based on horizontal position (0 for left half, 1 for right)
            col_index = int(x0 / (page_width / 2.0))
            # Sort primarily by column, then top-to-bottom (y0)
            return (col_index, y0)
            
        text_blocks.sort(key=block_sort_key)
        
        page_text = ""
        heading = None
        
        for b in text_blocks:
            block_text = b[4].strip()
            if not block_text:
                continue
                
            page_text += block_text + "\n\n"
            
            # Simple heading detection for the first substantial block
            if heading is None and len(block_text) < 100 and '\n' not in block_text.strip():
                # If it's title-cased or upper-cased, assume heading
                if block_text.isupper() or block_text.istitle():
                    heading = block_text

        if page_text.strip():
            pages_data.append({
                "page_num": page_num + 1,
                "text": page_text.strip(),
                "heading": heading,
                "blocks": len(text_blocks)
            })
            
    doc.close()
    return pages_data
