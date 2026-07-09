"""
scripts/setup_database.py
──────────────────────────
Run this script ONCE to initialise the RadLearn database.

What it does:
  1. Creates all data directories (data/, data/chroma/, data/documents/, data/images/)
  2. Executes sql/002_tables.sql  — creates all SQLite tables
  3. Executes sql/003_indexes.sql — creates all SQLite indexes
  4. Initialises ChromaDB collections (radlearn_chunks, radlearn_images)
  5. Inserts the default 'Manual Uploads' knowledge source

Usage:
    cd /path/to/radlearn-chatbot
    python scripts/setup_database.py

Safe to re-run: all CREATE TABLE and CREATE INDEX statements use IF NOT EXISTS.
ChromaDB uses get_or_create_collection so re-running will not duplicate collections.
"""

import sys
from pathlib import Path

# ── Add project root to sys.path so we can import radlearn.* ─────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from radlearn.config import DATA_DIR, CHROMA_DIR, DOCUMENTS_DIR, IMAGES_DIR, SQLITE_PATH
from radlearn.database.client import get_engine, get_chunks_collection, get_images_collection
from radlearn.database.documents import get_or_create_default_source

import sqlalchemy


def run_sql_file(file_path: Path, engine) -> None:
    """Execute all statements in a SQL file against the given engine."""
    sql = file_path.read_text()
    # Split on semicolons, filter blank statements
    statements = [s.strip() for s in sql.split(";") if s.strip()]

    with engine.begin() as conn:
        for stmt in statements:
            # Skip pure comments
            lines = [l for l in stmt.split("\n") if not l.strip().startswith("--")]
            clean = "\n".join(lines).strip()
            if clean:
                conn.execute(sqlalchemy.text(clean))


def main() -> None:
    print("=" * 55)
    print("  RadLearn — Database Setup")
    print("=" * 55)
    print()

    # ── Step 1: Directories ───────────────────────────────────────
    for d in [DATA_DIR, CHROMA_DIR, DOCUMENTS_DIR, IMAGES_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    print(f"✅  Data directories ready at: {DATA_DIR.resolve()}")

    # ── Step 2: SQLite tables + indexes ──────────────────────────
    sql_dir = ROOT / "sql"
    engine = get_engine()

    for filename in ["002_tables.sql", "003_indexes.sql", "004_projects.sql"]:
        file_path = sql_dir / filename
        if not file_path.exists():
            print(f"⚠️   {filename} not found — skipping")
            continue
        run_sql_file(file_path, engine)
        print(f"✅  {filename} executed")

    # ── Step 3: ChromaDB collections ─────────────────────────────
    chunks_col = get_chunks_collection()
    images_col = get_images_collection()
    print(f"✅  ChromaDB collection '{chunks_col.name}' ready")
    print(f"✅  ChromaDB collection '{images_col.name}' ready")

    # ── Step 4: Default knowledge source ─────────────────────────
    source = get_or_create_default_source()
    print(f"✅  Default knowledge source: '{source['name']}' (id: {source['id'][:8]}...)")

    # ── Summary ───────────────────────────────────────────────────
    print()
    print("=" * 55)
    print("  Setup complete!")
    print()
    print("  Next steps:")
    print("  1. Fill in your GOOGLE_API_KEY in .env")
    print("  2. Run:  streamlit run app.py")
    print("=" * 55)


if __name__ == "__main__":
    main()
