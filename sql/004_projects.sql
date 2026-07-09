-- sql/004_projects.sql
-- Adds the `projects` table and alters `conversations` and `documents` to link to it.

CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    system_instructions TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- SQLite ALTER TABLE does not easily support adding foreign key constraints, 
-- but we can add the column. In SQLite, foreign keys on newly added columns 
-- are not strictly enforced unless pragma foreign_keys=ON is set before the 
-- transaction, but it is sufficient for our app logic.

ALTER TABLE conversations ADD COLUMN project_id TEXT REFERENCES projects(id) ON DELETE SET NULL;
ALTER TABLE documents ADD COLUMN project_id TEXT REFERENCES projects(id) ON DELETE SET NULL;
