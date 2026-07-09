"""
radlearn/database/projects.py
──────────────────────────────
CRUD operations for Projects.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from radlearn.database.client import execute_sql

def create_project(name: str, system_instructions: str = "", description: str = "") -> dict:
    project_id = str(uuid.uuid4())
    record = {
        "id": project_id,
        "name": name,
        "description": description,
        "system_instructions": system_instructions,
        "created_at": _now()
    }
    
    execute_sql(
        """
        INSERT INTO projects (id, name, description, system_instructions, created_at)
        VALUES (:id, :name, :description, :system_instructions, :created_at)
        """,
        record
    )
    return record

def get_all_projects() -> list[dict]:
    return execute_sql("SELECT * FROM projects ORDER BY created_at DESC")

def get_project(project_id: str) -> dict | None:
    rows = execute_sql("SELECT * FROM projects WHERE id = :id", {"id": project_id})
    return rows[0] if rows else None

def delete_project(project_id: str) -> None:
    # First delete associated documents since they have foreign keys.
    # Note: deletion logic for documents should properly clean up Chroma and Filesystem.
    # For now, we rely on the admin panel or user manual deletion for documents, 
    # but the SQLite foreign key is SET NULL, so deleting a project just detaches documents.
    execute_sql("DELETE FROM projects WHERE id = :id", {"id": project_id})

def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
