#!/usr/bin/env python3
"""
Automated Answer Validation Runner
"""
import sys
import json
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from radlearn.chat.engine import process_query
from radlearn.chat.evaluator import calculate_grounding_metrics

def main():
    questions_file = Path(__file__).parent / "answer_validation_questions.json"
    with open(questions_file, "r") as f:
        dataset = json.load(f)
        
    print(f"Loaded {len(dataset)} questions for validation.")
    
    results = []
    
    # Metrics
    total_q = len(dataset)
    pass_count = 0
    citation_accurate_count = 0
    grounded_count = 0
    safe_negative_count = 0
    total_negatives = 0
    
    for i, item in enumerate(dataset):
        q = item["question"]
        expected = item["expected_type"]
        
        if expected == "negative":
            total_negatives += 1
            
        print(f"\n[{i+1}/{total_q}] Testing: {q}")
        try:
            res = process_query(q)
            
            if res.get("status") == "quota_exceeded":
                print(f"  TEST BLOCKED BY QUOTA: {res.get('message')}")
                results.append({"id": item["id"], "question": q, "pass": False, "error": "quota_exceeded"})
                continue
                
            eval_metrics = calculate_grounding_metrics(
                answer_text=res["answer"],
                citations=res["citations"],
                retrieved_chunks=res["retrieved_chunks"],
                confidence_score=res["confidence_score"]
            )
            
            # Pass Criteria
            # 1. Correct doc retrieved (Heuristic proxy: valid chunks returned)
            has_retrieved_docs = len(res["retrieved_chunks"]) > 0
            
            # 2. At least one valid citation
            has_valid_citation = len(res["citations"]) > 0
            
            # 3. No hallucinated citations
            no_hallucinations = eval_metrics["hallucination_risk"] == "LOW"
            
            # 4. Negative questions correctly refuse
            is_refusal = "information not available" in res["answer"].lower()
            
            passed = False
            
            if expected == "positive":
                if has_retrieved_docs and has_valid_citation and no_hallucinations:
                    passed = True
                    citation_accurate_count += 1
                    grounded_count += 1
            else: # negative
                if is_refusal and no_hallucinations:
                    passed = True
                    safe_negative_count += 1
                    
            if passed:
                pass_count += 1
                
            report_item = {
                "id": item["id"],
                "question": q,
                "expected_type": expected,
                "retrieved_docs": [c.get("doc_title", "") for c in res["retrieved_chunks"]],
                "answer_length": len(res["answer"]),
                "citation_count": len(res["citations"]),
                "hallucination_risk": eval_metrics["hallucination_risk"],
                "confidence_score": res["confidence_score"],
                "pass": passed
            }
            results.append(report_item)
            
            print(f"  Risk: {eval_metrics['hallucination_risk']} | Pass: {passed}")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"id": item["id"], "question": q, "pass": False, "error": str(e)})
            
        # Slight pause to respect any rate limits
        time.sleep(2)
        
    print("\n" + "="*50)
    print("  SUMMARY REPORT")
    print("="*50)
    
    answer_quality = (pass_count / total_q) * 100
    citation_accuracy = (citation_accurate_count / (total_q - total_negatives)) * 100 if (total_q - total_negatives) > 0 else 0
    grounding_score = (grounded_count / (total_q - total_negatives)) * 100 if (total_q - total_negatives) > 0 else 0
    negative_safety = (safe_negative_count / total_negatives) * 100 if total_negatives > 0 else 0
    
    print(f"Answer Quality Score      : {answer_quality:.1f}%")
    print(f"Citation Accuracy Score   : {citation_accuracy:.1f}%")
    print(f"Retrieval Grounding Score : {grounding_score:.1f}%")
    print(f"Negative Query Safety Score: {negative_safety:.1f}%")
    print("="*50)
    
    if answer_quality >= 80.0:
        print("\nREADY FOR STREAMLIT UI")
    else:
        print("\nFIX RAG BEFORE UI")

if __name__ == "__main__":
    main()
