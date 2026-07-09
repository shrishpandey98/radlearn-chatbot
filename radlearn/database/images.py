"""
radlearn/database/images.py
────────────────────────────
Image caption embedding storage and retrieval via ChromaDB.

Connection to system:
  - Called by ingestion/image_extractor.py after saving an image to disk.
  - Called by chat/engine.py to retrieve relevant images for each answer.
  - Images are stored locally at: data/images/{doc_id}/fig_NNN.png
  - ChromaDB stores: caption embedding + metadata (storage_path, doc_title, etc.)
  - The UI reads images using the storage_path from this metadata.
"""

from __future__ import annotations

from radlearn.database.client import get_images_collection
from radlearn.config import IMAGE_TOP_K


# ── Insert ────────────────────────────────────────────────────────

def add_image_to_chromadb(image: dict) -> None:
    """
    Insert a single image's caption embedding into ChromaDB.

    Args:
        image: dict with keys:
            id                UUID string
            document_id       UUID string
            doc_title         str
            source_url        str or None
            storage_path      str   local path e.g. "data/images/{doc_id}/fig_001.png"
            caption_text      str   text used for embedding
            caption_embedding list[float] of length 768
            figure_number     str or None
            page_number       int or None
            image_type        str  mri|ct|xray|ultrasound|pet|diagram|chart|photo|unknown
            specialty         str
            is_approved       bool
    """
    collection = get_images_collection()
    collection.add(
        ids=[image["id"]],
        embeddings=[image["caption_embedding"]],
        documents=[image.get("caption_text") or ""],
        metadatas=[{
            "document_id":   image["document_id"],
            "doc_title":     image.get("doc_title") or "",
            "source_url":    image.get("source_url") or "",
            "storage_path":  image["storage_path"],
            "figure_number": image.get("figure_number") or "",
            "page_number":   int(image.get("page_number") or -1),
            "image_type":    image.get("image_type") or "unknown",
            "specialty":     image.get("specialty") or "general",
            "is_approved":   bool(image.get("is_approved", True)),
        }],
    )


# ── Search ────────────────────────────────────────────────────────

def search_images_by_embedding(
    query_vector: list[float],
    top_k: int = IMAGE_TOP_K,
) -> list[dict]:
    """
    Retrieve the top-K most relevant images for a query vector.

    Args:
        query_vector: 768-dim embedding of the user query.
        top_k:        Maximum number of images to return.

    Returns:
        List of image dicts. Each dict contains:
            id, document_id, storage_path, caption_text,
            figure_number, image_type, doc_title, source_url, similarity_score
    """
    collection = get_images_collection()
    total = collection.count()
    if total == 0:
        return []

    try:
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k, total),
            where={"is_approved": True},
            include=["metadatas", "documents", "distances"],
        )
    except Exception:
        # Fallback without filter if no approved images exist
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k, total),
            include=["metadatas", "documents", "distances"],
        )

    images = []
    for i, image_id in enumerate(results["ids"][0]):
        meta = results["metadatas"][0][i]
        distance = results["distances"][0][i]
        similarity = round(1.0 - distance, 4)

        images.append({
            "id":               image_id,
            "document_id":      meta["document_id"],
            "storage_path":     meta["storage_path"],
            "caption_text":     results["documents"][0][i],
            "figure_number":    meta["figure_number"] or None,
            "image_type":       meta["image_type"],
            "doc_title":        meta["doc_title"],
            "source_url":       meta["source_url"] or None,
            "similarity_score": similarity,
        })

    return images


# ── Delete ────────────────────────────────────────────────────────

def delete_images_for_document(document_id: str) -> list[str]:
    """
    Delete all ChromaDB image entries for a document.

    Returns:
        List of storage_path strings for the caller to remove from disk.
    """
    collection = get_images_collection()
    try:
        existing = collection.get(
            where={"document_id": document_id},
            include=["metadatas"],
        )
    except Exception:
        return []

    storage_paths = [m["storage_path"] for m in existing["metadatas"]]
    if existing["ids"]:
        collection.delete(ids=existing["ids"])
    return storage_paths


def count_images() -> int:
    """Return total number of images in ChromaDB."""
    return get_images_collection().count()
