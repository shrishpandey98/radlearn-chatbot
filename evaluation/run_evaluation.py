"""
evaluation/run_evaluation.py
─────────────────────────────
Runs retrieval for all test questions and outputs results.
"""
import json
import sys
from pathlib import Path

# Ensure radlearn is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from radlearn.retrieval.preprocessor import preprocess_query
from radlearn.retrieval.searcher import hybrid_search
from radlearn.retrieval.ranker import reciprocal_rank_fusion

def run():
    questions_file = Path(__file__).parent / "test_questions.json"
    with open(questions_file, 'r') as f:
        questions = json.load(f)
        
    print(f"Running evaluation for {len(questions)} questions...\n")
    
    for q in questions:
        query = q["question"]
        print(f"[{q['specialty'].upper()}] Q: {query}")
        
        prep = preprocess_query(query)
        print(f"  Expanded: {prep['expanded_query']}")
        
        # Only test retrieval logic, not LLM
        search_res = hybrid_search(query, prep["expanded_query"], specialty_filter=q["specialty"], top_k=5)
        ranked = reciprocal_rank_fusion(search_res["semantic_results"], search_res["keyword_results"], top_n=3)
        
        if not ranked:
            print("  -> No chunks retrieved (Database empty?)")
        else:
            for i, chunk in enumerate(ranked):
                print(f"  [{i+1}] Score: {chunk['rrf_score']:.4f} | Doc: {chunk.get('doc_title', 'Unknown')} | Text: {chunk['text'][:60]}...")
        print("-" * 50)

if __name__ == "__main__":
    run()
