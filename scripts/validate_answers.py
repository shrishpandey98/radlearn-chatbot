#!/usr/bin/env python3
"""
Interactive End-to-End Answer Validation Tool
"""
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from radlearn.chat.engine import process_query
from radlearn.chat.evaluator import calculate_grounding_metrics

def main():
    print("========================================")
    print("  RadLearn End-to-End Validation Tool   ")
    print("========================================")
    print("Type 'exit' to quit.\n")
    
    while True:
        try:
            q = input("\nEnter Question > ")
            if q.lower().strip() in ('exit', 'quit'):
                break
            if not q.strip():
                continue
                
            start_time = time.time()
            res = process_query(q)
            latency = time.time() - start_time
            
            if res.get("status") == "quota_exceeded":
                print("\n" + "="*60)
                print("  VALIDATION RESULTS")
                print("="*60)
                print(f"Question         : {q}")
                print(f"Latency          : {latency:.2f}s")
                print("\n--- Error ---")
                print("TEST BLOCKED BY QUOTA")
                print(f"Message: {res.get('message')}")
                print("="*60)
                continue
                
            eval_metrics = calculate_grounding_metrics(
                answer_text=res["answer"],
                citations=res["citations"],
                retrieved_chunks=res["retrieved_chunks"],
                confidence_score=res["confidence_score"]
            )
            
            print("\n" + "="*60)
            print("  VALIDATION RESULTS")
            print("="*60)
            print(f"Question         : {q}")
            print(f"Expanded Query   : {res['expanded_query']}")
            print(f"Confidence Score : {res['confidence_score']:.4f}")
            print(f"Latency          : {latency:.2f}s")
            
            print("\n--- Retrieved Documents ---")
            docs = set()
            for chunk in res["retrieved_chunks"]:
                docs.add(chunk.get("doc_title", "Unknown Document"))
            for doc in docs:
                print(f" - {doc}")
                
            print("\n--- Final Prompt Sent to Gemini ---")
            print("SYSTEM:")
            print(res["prompt_sent"]["system"][:200] + "...\n")
            print("USER:")
            print(res["prompt_sent"]["user"][:300] + "...\n(truncated for brevity)")
            
            print("\n--- Gemini Response ---")
            print(res["answer"])
            
            print("\n--- Validated Citations ---")
            if not res["citations"]:
                print("None.")
            for c in res["citations"]:
                print(f"[{c['citation_number']}] {c['formatted_citation']} (Page {c['page_number']})")
                
            print("\n--- Hallucination Risk Evaluation ---")
            print(f"Risk Level : {eval_metrics['hallucination_risk']}")
            print(f"Reason     : {eval_metrics['reason']}")
            print(f"Metrics    : {eval_metrics['metrics']}")
            print("="*60)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error during validation: {e}")

if __name__ == "__main__":
    main()
