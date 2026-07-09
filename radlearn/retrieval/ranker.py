"""
radlearn/retrieval/ranker.py
─────────────────────────────
Implements Reciprocal Rank Fusion (RRF) to merge semantic and keyword results.
"""
from typing import List, Dict

def reciprocal_rank_fusion(semantic_results_list: List[List[Dict]], keyword_results_list: List[List[Dict]], k: int = 60, top_n: int = 10) -> List[Dict]:
    """
    Merges and ranks results using RRF.
    Expects lists of result lists (one list per extracted concept).
    Score = 1 / (k + rank)
    """
    rrf_scores = {}
    chunk_data = {}
    
    # Process semantic results
    for concept_results in semantic_results_list:
        for rank, chunk in enumerate(concept_results):
            chunk_id = chunk["id"]
            if chunk_id not in chunk_data:
                chunk_data[chunk_id] = chunk
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + (1.0 / (k + rank + 1))
        
    # Process keyword results
    for concept_results in keyword_results_list:
        for rank, chunk in enumerate(concept_results):
            chunk_id = chunk["id"]
            if chunk_id not in chunk_data:
                chunk_data[chunk_id] = chunk
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + (1.0 / (k + rank + 1))
        
    # Sort by RRF score descending
    ranked_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Calculate max similarity score for filtering
    max_sim = 0.0
    for chunk in chunk_data.values():
        sim = chunk.get("similarity_score", 0.0)
        if sim > max_sim:
            max_sim = sim
            
    sim_threshold = 0.9 * max_sim
    
    # Build final list, inject rrf_score, enforce document diversity, and filter by similarity
    final_results = []
    doc_counts = {}
    
    for chunk_id, score in ranked_chunks:
        if len(final_results) >= top_n:
            break
            
        chunk = chunk_data[chunk_id].copy()
        
        # Filter by similarity threshold
        chunk_sim = chunk.get("similarity_score", 0.0)
        if chunk_sim < sim_threshold:
            continue
            
        doc_key = chunk.get("doc_title", chunk.get("id"))
        
        # Max 3 chunks per document
        if doc_counts.get(doc_key, 0) >= 3:
            continue
            
        doc_counts[doc_key] = doc_counts.get(doc_key, 0) + 1
        chunk["rrf_score"] = score
        final_results.append(chunk)
        
    return final_results
