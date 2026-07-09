"""
radlearn/retrieval/searcher.py
───────────────────────────────
Hybrid search execution calling semantic and keyword searches.
"""
from radlearn.database.chunks import semantic_search, keyword_search
from radlearn.ingestion.embedder import embed_single

def hybrid_search(query: str, expanded_query: str, specialty_filter: str = None, top_k: int = 15, session_id: str = None, project_id: str = None) -> dict:
    """
    Executes both semantic and keyword searches.
    """
    # Embed the expanded query for better semantic matching
    query_vector = embed_single(expanded_query)
    
    semantic_results = semantic_search(
        query_vector=query_vector, 
        specialty_filter=specialty_filter, 
        top_k=top_k,
        session_id=session_id,
        project_id=project_id
    )
    
    # Keyword search uses expanded query to match exact terms
    keyword_results = keyword_search(
        query_text=expanded_query, 
        specialty_filter=specialty_filter, 
        top_k=top_k,
        session_id=session_id,
        project_id=project_id
    )
    
    return {
        "semantic_results": semantic_results,
        "keyword_results": keyword_results
    }
