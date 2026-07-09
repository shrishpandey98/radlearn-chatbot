"""
scripts/migrate_projects.py
────────────────────────────
Applies the 004_projects.sql schema changes to add the Projects feature.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from radlearn.database.client import get_engine
import sqlalchemy

def main():
    engine = get_engine()
    sql_path = ROOT / "sql" / "004_projects.sql"
    
    if not sql_path.exists():
        print(f"Error: {sql_path} not found.")
        sys.exit(1)
        
    sql = sql_path.read_text()
    statements = [s.strip() for s in sql.split(";") if s.strip()]
    
    with engine.begin() as conn:
        for stmt in statements:
            lines = [l for l in stmt.split("\n") if not l.strip().startswith("--")]
            clean = "\n".join(lines).strip()
            if clean:
                try:
                    conn.execute(sqlalchemy.text(clean))
                    print(f"✅ Executed: {clean[:50]}...")
                except sqlalchemy.exc.OperationalError as e:
                    if "duplicate column name" in str(e).lower():
                        print(f"⚠️ Column already exists, skipping: {clean[:50]}...")
                    else:
                        print(f"❌ Error executing: {clean[:50]}...")
                        print(e)
                        raise

    print("\n✅ Database migration for Projects feature complete!")

if __name__ == "__main__":
    main()
