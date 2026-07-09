"""
radlearn/ingestion/image_extractor.py
─────────────────────────────────────
Extracts images from PDFs using PyMuPDF, filters out tiny graphics/logos,
finds nearby captions, and stores embeddings in ChromaDB.
"""
import logging
import uuid
import fitz
import io
from pathlib import Path
from radlearn.config import IMG_MIN_WIDTH, IMG_MIN_HEIGHT, IMG_MIN_SIZE_BYTES, IMAGES_DIR
from radlearn.ingestion.embedder import embed_single
from radlearn.database.images import add_image_to_chromadb

logger = logging.getLogger(__name__)

def extract_images_from_pdf(file_path: str, document_id: str, doc_metadata: dict) -> int:
    """
    Extracts images from PDF, links captions, embeds them, and saves to ChromaDB.
    Returns the number of images successfully extracted and stored.
    """
    try:
        doc = fitz.open(file_path)
    except Exception as e:
        logger.error(f"Failed to open PDF for image extraction {file_path}: {e}")
        return 0

    img_dir = IMAGES_DIR / document_id
    img_dir.mkdir(parents=True, exist_ok=True)
    
    extracted_count = 0

    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images(full=True)
        
        # Get text blocks to find captions near images
        blocks = page.get_text("blocks")
        text_blocks = [b for b in blocks if b[6] == 0]

        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            width = base_image["width"]
            height = base_image["height"]
            ext = base_image["ext"]
            
            # Filter out small logos/icons
            if width < IMG_MIN_WIDTH or height < IMG_MIN_HEIGHT or len(image_bytes) < IMG_MIN_SIZE_BYTES:
                continue

            # Attempt to find bounding box of image on page to locate caption
            # This is a heuristic. PyMuPDF get_image_rects can fail or return multiple rects.
            try:
                rects = page.get_image_rects(xref)
                if rects:
                    img_rect = rects[0]
                    caption_text = _find_caption_near_rect(img_rect, text_blocks)
                else:
                    caption_text = "Radiology figure"
            except Exception:
                caption_text = "Radiology figure"

            # If the caption is too generic, it might not retrieve well, but we embed what we have.
            caption_embedding = embed_single(caption_text)
            if not caption_embedding:
                continue

            # Save image to disk
            filename = f"fig_p{page_num+1}_{img_index+1}.{ext}"
            file_path_local = img_dir / filename
            file_path_local.write_bytes(image_bytes)
            
            # Extract Figure Number if present
            fig_num = ""
            if caption_text.lower().startswith("fig"):
                words = caption_text.split()
                if len(words) >= 2:
                    fig_num = words[0] + " " + words[1].strip(":")

            image_record = {
                "id": str(uuid.uuid4()),
                "document_id": document_id,
                "doc_title": doc_metadata.get("title", ""),
                "source_url": doc_metadata.get("source_url", ""),
                "storage_path": str(file_path_local),
                "caption_text": caption_text,
                "caption_embedding": caption_embedding,
                "figure_number": fig_num,
                "page_number": page_num + 1,
                "image_type": "unknown", # Advanced logic could detect MRI vs CT here
                "specialty": doc_metadata.get("specialty", "general"),
                "is_approved": True
            }
            
            add_image_to_chromadb(image_record)
            extracted_count += 1

    doc.close()
    return extracted_count

def _find_caption_near_rect(img_rect: fitz.Rect, text_blocks: list) -> str:
    """
    Finds the text block immediately below the image rectangle.
    """
    closest_block = None
    min_dist = float('inf')
    
    for b in text_blocks:
        x0, y0, x1, y1, text, _, _ = b
        
        # Check if text block is below the image
        if y0 >= img_rect.y1 - 10: # Allow slight overlap
            dist = y0 - img_rect.y1
            # Must be horizontally aligned somewhat
            if x1 > img_rect.x0 and x0 < img_rect.x1:
                if dist < min_dist and dist < 100: # Max 100pts away
                    min_dist = dist
                    closest_block = text.strip()
                    
    # Often captions start with Fig or Figure
    if closest_block and (closest_block.lower().startswith("fig") or closest_block.lower().startswith("table")):
        return closest_block.replace("\n", " ")
        
    return closest_block.replace("\n", " ") if closest_block else "Radiology figure"
