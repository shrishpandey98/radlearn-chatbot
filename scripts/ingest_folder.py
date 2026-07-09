"""
scripts/ingest_folder.py
─────────────────────────
Batch ingestion utility. Processes all PDFs and DOCX files in a folder.
"""
import sys
import time
from pathlib import Path

# Ensure radlearn is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from radlearn.ingestion.pipeline import ingest_file

# Seconds to wait between documents to avoid hitting the free-tier rate limit.
# Google AI Studio: 1,500 embedding requests/minute on free tier.
# A large PDF can use ~50 requests. 12s between docs keeps us well under the cap.
INTER_DOCUMENT_PAUSE = 30

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/ingest_folder.py <folder_path>")
        sys.exit(1)
        
    folder_path = Path(sys.argv[1])
    if not folder_path.is_dir():
        print(f"Error: {folder_path} is not a valid directory.")
        sys.exit(1)
        
    print(f"Scanning {folder_path} for PDFs and DOCX files...")
    
    files_to_process = list(folder_path.rglob("*.pdf")) + list(folder_path.rglob("*.docx"))
    
    if not files_to_process:
        print("No supported files found.")
        sys.exit(0)
        
    print(f"Found {len(files_to_process)} files to process.")
    print(f"Note: {INTER_DOCUMENT_PAUSE}s pause between documents to respect API rate limits.\n")
    
    results = {"success": 0, "skipped": 0, "failed": 0, "chunks": 0, "images": 0}
    start_time = time.time()
    
    for i, filepath in enumerate(files_to_process):
        print(f"[{i+1}/{len(files_to_process)}] Processing: {filepath.name}")
        
        file_bytes = filepath.read_bytes()
        metadata = {
            "title": filepath.stem,
            "doc_type": "article",
            "specialty": "general",
            "license_type": "internal",
            "knowledge_source_id": None # Will default to Manual Uploads
        }
        
        res = ingest_file(file_bytes, filepath.name, metadata)
        
        if res["status"] == "success":
            results["success"] += 1
            results["chunks"] += res["chunks"]
            results["images"] += res.get("images", 0)
            print(f"  -> Success: {res['chunks']} chunks, {res.get('images', 0)} images.")
        elif res["status"] == "skipped":
            results["skipped"] += 1
            print(f"  -> Skipped: {res['reason']}")
        else:
            results["failed"] += 1
            print(f"  -> Failed: {res['reason']}")
            
        # Pause between documents (skip after last file, skip for duplicates)
        if i < len(files_to_process) - 1 and res["status"] != "skipped":
            for remaining in range(INTER_DOCUMENT_PAUSE, 0, -1):
                print(f"  [Rate limit pause: {remaining}s remaining...]", end="\r")
                time.sleep(1)
            print(" " * 40, end="\r")  # Clear the countdown line
            
    elapsed = time.time() - start_time
    
    print("\n" + "="*42)
    print("  Ingestion Report")
    print("="*42)
    print(f"  Total time : {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Processed  : {results['success']}")
    print(f"  Skipped    : {results['skipped']} (duplicates)")
    print(f"  Failed     : {results['failed']}")
    print(f"  Chunks     : {results['chunks']}")
    print(f"  Images     : {results['images']}")
    print("="*42)

if __name__ == "__main__":
    main()
