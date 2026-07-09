"""
radlearn/chat/evaluator.py
──────────────────────────
Heuristic hallucination detection and answer evaluation.
"""
import re
import json
from typing import List, Dict
from radlearn.chat.llm_factory import get_llm_client

HALLUCINATION_PROMPT = """You are an expert medical hallucination detector.
Review the provided Answer against the provided Context Chunks and the original Question.
Identify any diagnoses, risk factors, or patient characteristics in the Answer that are NOT present in either the Question or the Context Chunks.
Respond with a JSON object with two fields:
1. "hallucinations_found": boolean
2. "details": string explaining what was unsupported (or "None" if none)

Context:
{context}

Question:
{question}

Answer:
{answer}

Output ONLY valid JSON.
"""

def detect_semantic_hallucination(answer_text: str, retrieved_chunks: List[Dict], question: str = "") -> dict:
    context_str = "\n".join([f"Chunk {i+1}: {c.get('text', '')}" for i, c in enumerate(retrieved_chunks)])
    prompt = HALLUCINATION_PROMPT.format(context=context_str, question=question, answer=answer_text)
    
    try:
        client = get_llm_client()
        raw_response = client.generate_answer("You are a helpful JSON API.", prompt)
        
        start = raw_response.find("{")
        end = raw_response.rfind("}") + 1
        if start != -1 and end != 0:
            json_str = raw_response[start:end]
            result = json.loads(json_str)
            return {
                "hallucinations_found": result.get("hallucinations_found", False),
                "details": result.get("details", "")
            }
        else:
            return {"hallucinations_found": False, "details": "Failed to parse"}
    except Exception as e:
        return {"hallucinations_found": False, "details": f"Error: {str(e)}"}

def calculate_grounding_metrics(answer_text: str, citations: List[Dict], retrieved_chunks: List[Dict], confidence_score: float, question: str = "") -> Dict:
    """
    Evaluates the hallucination risk of an answer using heuristics.
    """
    if not retrieved_chunks:
        # If no chunks were retrieved, the answer should be a refusal.
        if "information not available" in answer_text.lower() or "cannot answer" in answer_text.lower():
            return {"hallucination_risk": "LOW", "reason": "Correctly refused due to missing context."}
        else:
            return {"hallucination_risk": "HIGH", "reason": "Answered despite no retrieved context."}

    # 1. Check for unsupported citation markers in the text
    # Find all [N] in text
    all_markers = set(int(m) for m in re.findall(r'\[(\d+)\]', answer_text))
    valid_citation_nums = set(c["citation_number"] for c in citations)
    
    unsupported_markers = all_markers - valid_citation_nums
    
    # 2. Chunk utilization
    utilized_ratio = len(valid_citation_nums) / len(retrieved_chunks) if retrieved_chunks else 0
    
    # 3. Grounding coverage (proxy: percentage of answer length that is close to citations)
    # Simple heuristic: Does the answer use citations? 
    has_citations = len(citations) > 0
    
    # Determine Risk
    risk = "LOW"
    reasons = []
    
    if unsupported_markers:
        risk = "HIGH"
        reasons.append(f"Contains unsupported citation markers: {unsupported_markers}")
        
    if not has_citations:
        if "information not available" not in answer_text.lower():
            risk = "HIGH"
            reasons.append("Answer provided without any valid citations.")
            
    if confidence_score < 0.015: # Arbitrary low confidence threshold
        if risk == "LOW":
            risk = "MEDIUM"
            reasons.append(f"Low retrieval confidence ({confidence_score:.4f}).")
            
    if risk == "LOW":
        if question:
            sem_eval = detect_semantic_hallucination(answer_text, retrieved_chunks, question)
            if sem_eval["hallucinations_found"]:
                risk = "HIGH"
                reasons.append(f"Unsupported semantics found: {sem_eval['details']}")
            else:
                reasons.append("All citations are valid and supported.")
        else:
            reasons.append("All citations are valid and supported.")
        
    return {
        "hallucination_risk": risk,
        "reason": " ".join(reasons),
        "metrics": {
            "valid_citations": len(valid_citation_nums),
            "unsupported_markers": len(unsupported_markers),
            "chunk_utilization": utilized_ratio,
            "confidence_score": confidence_score
        }
    }
