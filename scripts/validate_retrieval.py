#!/usr/bin/env python3
"""
scripts/validate_retrieval.py
──────────────────────────────
Interactive CLI tool to validate retrieval quality BEFORE building the UI.
Runs the full retrieval pipeline (preprocess → hybrid search → RRF rank)
WITHOUT calling Gemini. Pure retrieval-only validation.

Usage:
    python3 scripts/validate_retrieval.py
"""
import sys
import time
from pathlib import Path
from collections import Counter

# ── Path setup ────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from radlearn.retrieval.preprocessor import preprocess_query
from radlearn.retrieval.searcher import hybrid_search
from radlearn.retrieval.ranker import reciprocal_rank_fusion
from radlearn.database.chunks import count_chunks

# ── ANSI colours (work on macOS Terminal and iTerm2) ─────────────
RESET   = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
CYAN    = "\033[36m"
GREEN   = "\033[32m"
YELLOW  = "\033[33m"
RED     = "\033[31m"
MAGENTA = "\033[35m"
BLUE    = "\033[34m"
WHITE   = "\033[97m"

# ── Configuration ─────────────────────────────────────────────────
TOP_K_SEARCH  = 15   # candidates from each search leg
TOP_N_RANKED  = 10   # final ranked chunks to display
PREVIEW_CHARS = 180  # characters of chunk text to preview


# ──────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────

def hr(char="─", width=72, colour=DIM):
    print(f"{colour}{char * width}{RESET}")


def score_bar(score: float, width: int = 20) -> str:
    """Visual bar representing an RRF score (0 → max ~0.033)."""
    # RRF max for top-1 in both lists: 1/(60+1) + 1/(60+1) ≈ 0.033
    filled = int(min(score / 0.033, 1.0) * width)
    bar = "█" * filled + "░" * (width - filled)
    if score > 0.020:
        colour = GREEN
    elif score > 0.010:
        colour = YELLOW
    else:
        colour = RED
    return f"{colour}{bar}{RESET}"


def confidence_label(score: float) -> str:
    if score > 0.020:
        return f"{GREEN}HIGH{RESET}"
    elif score > 0.010:
        return f"{YELLOW}MEDIUM{RESET}"
    else:
        return f"{RED}LOW{RESET}"


def truncate(text: str, n: int = PREVIEW_CHARS) -> str:
    text = text.replace("\n", " ").strip()
    return text[:n] + "…" if len(text) > n else text


def compute_chunk_diversity(chunks: list) -> dict:
    """
    Returns stats about how many unique documents and specialties appear
    in the top ranked results — a proxy for answer breadth.
    """
    doc_titles = [c.get("doc_title", "Unknown") for c in chunks]
    specialties = [c.get("doc_specialty", "general") for c in chunks]
    unique_docs = list(dict.fromkeys(doc_titles))   # preserve order, deduplicate
    return {
        "unique_doc_count":  len(set(doc_titles)),
        "unique_docs":       unique_docs,
        "specialty_counts":  dict(Counter(specialties)),
        "chunk_total":       len(chunks),
    }


# ──────────────────────────────────────────────────────────────────
# Core retrieval runner
# ──────────────────────────────────────────────────────────────────

def run_retrieval(question: str) -> dict:
    """
    Full pipeline: preprocess → hybrid search → RRF rank.
    Returns timing metrics and ranked results. Does NOT call Gemini.
    """
    # 1. Preprocess ──────────────────────────────────────────────
    t0 = time.perf_counter()
    prep = preprocess_query(question)
    preprocess_ms = (time.perf_counter() - t0) * 1000

    # 2. Hybrid Search ──────────────────────────────────────────
    t1 = time.perf_counter()
    search_results = hybrid_search(
        query=question,
        expanded_query=prep["expanded_query"],
        specialty_filter=None,   # let the preprocessor detected specialty inform RRF only
        top_k=TOP_K_SEARCH,
    )
    search_ms = (time.perf_counter() - t1) * 1000

    # 3. RRF Ranking ─────────────────────────────────────────────
    t2 = time.perf_counter()
    ranked = reciprocal_rank_fusion(
        semantic_results=search_results["semantic_results"],
        keyword_results=search_results["keyword_results"],
        top_n=TOP_N_RANKED,
    )
    rank_ms = (time.perf_counter() - t2) * 1000

    total_ms = (time.perf_counter() - t0) * 1000

    return {
        "question":         question,
        "expanded_query":   prep["expanded_query"],
        "specialty":        prep["specialty"],
        "semantic_hits":    len(search_results["semantic_results"]),
        "keyword_hits":     len(search_results["keyword_results"]),
        "ranked_chunks":    ranked,
        "preprocess_ms":    preprocess_ms,
        "search_ms":        search_ms,
        "rank_ms":          rank_ms,
        "total_ms":         total_ms,
    }


# ──────────────────────────────────────────────────────────────────
# Display
# ──────────────────────────────────────────────────────────────────

def display_results(result: dict) -> None:
    ranked    = result["ranked_chunks"]
    diversity = compute_chunk_diversity(ranked)
    top_score = ranked[0]["rrf_score"] if ranked else 0.0

    print()
    hr("═", colour=CYAN)
    print(f"{BOLD}{CYAN}  RETRIEVAL RESULTS{RESET}")
    hr("═", colour=CYAN)

    # Query info ─────────────────────────────────────────────────
    print(f"\n{BOLD}  Question :{RESET} {result['question']}")
    if result["expanded_query"] != result["question"]:
        print(f"  {DIM}Expanded  : {result['expanded_query']}{RESET}")
    print(f"  Specialty : {MAGENTA}{result['specialty']}{RESET}")
    print(f"  Semantic  : {result['semantic_hits']} candidates  |  "
          f"Keyword : {result['keyword_hits']} candidates")

    # Chunks ─────────────────────────────────────────────────────
    print()
    hr()
    if not ranked:
        print(f"\n  {RED}No chunks retrieved. Knowledge base may be empty.{RESET}\n")
        return

    for i, chunk in enumerate(ranked):
        rank_num    = i + 1
        rrf         = chunk["rrf_score"]
        sem_score   = chunk.get("similarity_score", 0.0)
        title       = chunk.get("doc_title", "Unknown")[:55]
        page        = chunk.get("page_number")
        heading     = chunk.get("section_heading") or "—"
        specialty   = chunk.get("doc_specialty", "general")

        page_str = f"p.{page}" if page else "n/a"

        print(f"\n  {BOLD}[{rank_num:02d}]{RESET} {score_bar(rrf)}  RRF {BOLD}{rrf:.4f}{RESET}")
        print(f"       {BOLD}Source:{RESET}  {WHITE}{title}{RESET}")
        print(f"       {BOLD}Page:{RESET}    {page_str}   "
              f"{BOLD}Heading:{RESET} {DIM}{heading[:50]}{RESET}")
        print(f"       {BOLD}Specialty:{RESET} {specialty}   "
              f"{BOLD}Sem. score:{RESET} {sem_score:.4f}")
        print(f"       {DIM}{truncate(chunk['text'])}{RESET}")
        if i < len(ranked) - 1:
            hr("·", colour=DIM)

    # Latency ────────────────────────────────────────────────────
    print()
    hr()
    print(f"\n  {BOLD}⏱  Latency{RESET}")
    print(f"     Preprocess : {result['preprocess_ms']:>7.1f} ms")
    print(f"     Search     : {result['search_ms']:>7.1f} ms  "
          f"{DIM}(embed query + ChromaDB + FTS5){RESET}")
    print(f"     Rank (RRF) : {result['rank_ms']:>7.1f} ms")
    print(f"     {BOLD}Total      : {result['total_ms']:>7.1f} ms{RESET}")

    # Summary ────────────────────────────────────────────────────
    print()
    hr()
    print(f"\n  {BOLD}📊  Retrieval Summary{RESET}")
    print(f"     Chunks retrieved    : {diversity['chunk_total']}")
    print(f"     Unique documents    : {diversity['unique_doc_count']}")
    print(f"     Top confidence      : {confidence_label(top_score)}  ({top_score:.4f})")
    print(f"     Specialty breakdown : {diversity['specialty_counts']}")
    print()
    print(f"  {BOLD}  Source documents in results:{RESET}")
    for j, doc in enumerate(diversity["unique_docs"]):
        print(f"     {j+1}. {doc[:65]}")

    print()
    hr("═", colour=CYAN)
    print()


# ──────────────────────────────────────────────────────────────────
# Main interactive loop
# ──────────────────────────────────────────────────────────────────

def main():
    # Startup banner ─────────────────────────────────────────────
    print()
    hr("═", colour=CYAN)
    print(f"{BOLD}{CYAN}  RadLearn — Retrieval Validation CLI{RESET}")
    hr("═", colour=CYAN)

    # Quick DB health check
    try:
        total = count_chunks()
    except Exception as e:
        print(f"\n{RED}  ERROR: Cannot connect to ChromaDB: {e}{RESET}")
        print(f"  Run  python3 scripts/setup_database.py  first.\n")
        sys.exit(1)

    if total == 0:
        print(f"\n{YELLOW}  WARNING: ChromaDB is empty. No chunks found.{RESET}")
        print(f"  Run  python3 scripts/ingest_folder.py <folder>  first.\n")
    else:
        print(f"\n  {GREEN}✓  ChromaDB ready — {total} chunks indexed{RESET}")

    print(f"\n  {DIM}Commands:  type your question and press Enter{RESET}")
    print(f"  {DIM}           'quit' or 'exit' to stop{RESET}")
    print(f"  {DIM}           'stats' to show DB summary{RESET}")
    print()

    session_count = 0
    session_latencies = []

    while True:
        # Prompt ─────────────────────────────────────────────────
        try:
            hr("─", colour=DIM)
            raw = input(f"{BOLD}{CYAN}  Query > {RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n\n  {DIM}Session ended.{RESET}\n")
            break

        if not raw:
            continue

        if raw.lower() in ("quit", "exit", "q"):
            break

        if raw.lower() == "stats":
            _show_db_stats()
            continue

        # Run retrieval ──────────────────────────────────────────
        print(f"\n  {DIM}Running retrieval...{RESET}", end="\r")
        try:
            result = run_retrieval(raw)
        except Exception as e:
            print(f"\n  {RED}Retrieval error: {e}{RESET}\n")
            continue

        display_results(result)

        session_count += 1
        session_latencies.append(result["total_ms"])

    # Session summary ────────────────────────────────────────────
    if session_count > 0:
        avg_lat = sum(session_latencies) / len(session_latencies)
        print(f"  {BOLD}Session Summary:{RESET}  "
              f"{session_count} queries   |   "
              f"avg latency: {avg_lat:.0f} ms\n")


def _show_db_stats():
    """Print a quick database snapshot."""
    from radlearn.database.documents import get_document_stats
    stats = get_document_stats()
    total_chunks = count_chunks()
    print()
    hr()
    print(f"  {BOLD}Database Snapshot{RESET}")
    print(f"     Documents completed : {stats.get('completed', 0)}")
    print(f"     Documents failed    : {stats.get('failed', 0)}")
    print(f"     Total chunks (SQL)  : {stats.get('total_chunks', 0)}")
    print(f"     ChromaDB vectors    : {total_chunks}")
    print(f"     Total images        : {stats.get('total_images', 0)}")
    hr()
    print()


if __name__ == "__main__":
    main()
