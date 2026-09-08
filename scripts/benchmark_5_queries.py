import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from radlearn.retrieval.preprocessor import preprocess_query
from radlearn.retrieval.searcher import hybrid_search
from radlearn.retrieval.ranker import reciprocal_rank_fusion

queries = [
    "What does BI-RADS 4 indicate?",
    "What supplemental screening is recommended for dense breasts?",
    "What imaging modality is preferred for plexopathy?",
    "What imaging is appropriate for suspected acute aortic syndrome?",
    "What imaging is recommended for suspected spine infection?"
]

def run_benchmark():
    for i, q in enumerate(queries):
        print(f"\n--- Q{i+1}: {q} ---")
        prep = preprocess_query(q)
        
        try:
            results = hybrid_search(
                query=prep["original_query"],
                expanded_query=prep["expanded_query"],
                specialty_filter=prep["specialty"],
                top_k=15
            )
            
            ranked = reciprocal_rank_fusion(
                semantic_results_list=[results["semantic_results"]],
                keyword_results_list=[results["keyword_results"]]
            )
            
            for j, chunk in enumerate(ranked[:5]):
                doc_title = chunk.get("doc_title", "UNKNOWN")
                page_number = chunk.get("page_number", "?")
                semantic_score = chunk.get("similarity_score", 0.0)
                keyword_score = chunk.get("keyword_score", 0.0)
                rrf_score = chunk.get("rrf_score", 0.0)
                text = chunk.get("text", "")[:100].replace("\n", " ")
                
                print(f"[{j+1}] doc={doc_title} (p.{page_number}) sem={semantic_score:.4f} key={keyword_score:.4f} rrf={rrf_score:.4f}")
                print(f"    {text}...")
                
        except Exception as e:
            print(f"Failed to retrieve: {e}")

if __name__ == "__main__":
    run_benchmark()
