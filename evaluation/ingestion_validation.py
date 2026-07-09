"""
evaluation/ingestion_validation.py
───────────────────────────────────
Validates that ingestion properly populated databases and
makes chunks available for retrieval.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from radlearn.database.documents import get_document_stats
from radlearn.database.chunks import count_chunks
from radlearn.retrieval.searcher import hybrid_search

def run_validation():
    print("Running Ingestion Validation...\n")
    
    stats = get_document_stats()
    chroma_count = count_chunks()
    
    print("Database State:")
    print(f"  Documents processed : {stats['completed']}")
    print(f"  Total chunks (SQL)  : {stats['total_chunks']}")
    print(f"  Total images        : {stats['total_images']}")
    print(f"  ChromaDB vectors    : {chroma_count}")
    
    if stats['completed'] == 0:
        print("\nNo documents have been ingested yet. Run ingest_folder.py first.")
        return
        
    if stats['total_chunks'] != chroma_count:
        print("\n[WARNING] SQLite chunk count does not match ChromaDB vector count!")
    else:
        print("\n[OK] Database synchronization looks good.")
        
    print("\nTesting Retrieval Visibility...")
    # Just run a generic search that should match something if db isn't empty
    res = hybrid_search("radiology OR imaging", "radiology OR imaging", top_k=1)
    
    if res["semantic_results"]:
        print("[OK] Semantic search successfully retrieved chunks.")
    else:
        print("[FAIL] Semantic search returned nothing.")
        
    if res["keyword_results"]:
        print("[OK] Keyword search successfully retrieved chunks.")
    else:
        print("[FAIL] Keyword search returned nothing.")

if __name__ == "__main__":
    run_validation()
