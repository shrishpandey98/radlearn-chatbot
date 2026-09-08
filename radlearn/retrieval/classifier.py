"""
radlearn/retrieval/classifier.py
─────────────────────────────────
Analyzes user queries to determine question type and extract independent concepts.
Optimized with fast heuristics to avoid slow blocking LLM roundtrips during retrieval.
"""
import re

def classify_query(query: str) -> dict:
    """
    Fast heuristic classifier to eliminate redundant LLM latency (<1ms).
    Extracts independent concepts when comparisons or differentials are asked.
    """
    if not query:
        return {"type": "factual", "concepts": []}
        
    lower = query.lower().strip()
    
    # 1. Comparison detection
    comp_delimiters = [r'\bvs\.?\b', r'\bversus\b', r'\bcompared to\b', r'\bdifference between\b']
    for pat in comp_delimiters:
        if re.search(pat, lower):
            parts = [p.strip() for p in re.split(pat, query, flags=re.IGNORECASE) if p.strip()]
            if len(parts) >= 2:
                return {
                    "type": "comparison",
                    "concepts": parts
                }
                
    # 2. Differential diagnosis detection
    diff_keywords = ["differential", "ddx", "differentials", "possible causes", "etiology", "etiologies"]
    if any(k in lower for k in diff_keywords):
        return {
            "type": "differential",
            "concepts": [query.strip()]
        }
        
    # 3. Recommendation / guidelines detection
    rec_keywords = ["appropriate", "indication", "recommend", "guideline", "criteria", "next step", "management"]
    if any(k in lower for k in rec_keywords):
        return {
            "type": "recommendation",
            "concepts": [query.strip()]
        }
        
    # 4. Explanatory
    if lower.startswith(("why ", "how ", "explain")):
        return {
            "type": "explanatory",
            "concepts": [query.strip()]
        }
        
    # 5. Default factual
    return {
        "type": "factual",
        "concepts": [query.strip()]
    }

