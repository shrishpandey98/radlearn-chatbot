"""
radlearn/retrieval/classifier.py
─────────────────────────────────
Analyzes user queries to determine question type and extract independent concepts.
"""
import json
from radlearn.chat.llm_factory import get_llm_client

CLASSIFIER_PROMPT = """You are an expert radiology query analyzer.
Your task is to analyze a user's question and output a JSON object with two fields:
1. "type": The type of question. Must be one of ["factual", "explanatory", "comparison", "differential", "recommendation"].
2. "concepts": A list of independent concepts, diseases, imaging modalities, or procedures found in the query. If the query asks to compare or differentiate, list each entity separately. 
CRITICAL: Do NOT extract generic terms like "imaging modality", "diagnostic information", "patient", "treatment", or "preferred". Only extract specific medical conditions, anatomies, specific modalities (e.g. "MRI", "CT"), or procedures.

Example 1:
Question: "adrenal mass vs chronic cough"
Output:
{"type": "comparison", "concepts": ["adrenal mass", "chronic cough"]}

Example 2:
Question: "What are the key findings of MS on an MRI?"
Output:
{"type": "factual", "concepts": ["multiple sclerosis", "MRI"]}

Example 3:
Question: "CTA vs MRA in vasculitis"
Output:
{"type": "comparison", "concepts": ["CTA", "MRA", "vasculitis"]}

Output ONLY valid JSON.
"""

def classify_query(query: str) -> dict:
    """
    Classifies the question and extracts independent concepts using the LLM.
    """
    try:
        client = get_llm_client()
        raw_response = client.generate_answer(CLASSIFIER_PROMPT, f"Question: {query}")
        
        # Parse JSON
        start = raw_response.find("{")
        end = raw_response.rfind("}") + 1
        if start != -1 and end != 0:
            json_str = raw_response[start:end]
            result = json.loads(json_str)
            # Ensure concepts is not empty
            concepts = result.get("concepts", [])
            if not concepts:
                concepts = [query]
            return {
                "type": result.get("type", "factual"),
                "concepts": concepts
            }
        else:
            return {"type": "factual", "concepts": [query]}
    except Exception as e:
        return {"type": "factual", "concepts": [query]}
