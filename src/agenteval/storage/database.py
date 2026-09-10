from __future__ import annotations

from pathlib import Path
import sqlite3


SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS datasets (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS dataset_cases (
    id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    input_json TEXT NOT NULL,
    expected_json TEXT,
    metadata_json TEXT,
    FOREIGN KEY(dataset_id) REFERENCES datasets(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    dataset_id TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS experiment_variants (
    id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    name TEXT NOT NULL,
    agent_ref TEXT,
    metadata_json TEXT,
    FOREIGN KEY(experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    experiment_id TEXT,
    dataset_case_id TEXT NOT NULL,
    input_json TEXT NOT NULL,
    output_json TEXT,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    duration REAL,
    token_usage_json TEXT,
    estimated_cost REAL,
    metadata_json TEXT,
    trace_id TEXT
);

CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    parent_id TEXT,
    type TEXT NOT NULL,
    name TEXT,
    timestamp_start TEXT NOT NULL,
    timestamp_end TEXT,
    input_json TEXT,
    output_json TEXT,
    metadata_json TEXT,
    status TEXT NOT NULL,
    error TEXT,
    FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS evaluators (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    kind TEXT,
    metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS evaluation_results (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    evaluator TEXT NOT NULL,
    score REAL,
    passed INTEGER,
    explanation TEXT,
    status TEXT NOT NULL,
    metadata_json TEXT,
    evidence_json TEXT,
    error TEXT,
    FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS failure_classifications (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    failure_type TEXT NOT NULL,
    likely_root_cause TEXT,
    confidence REAL,
    evidence_json TEXT,
    metadata_json TEXT,
    FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS inbox_entries (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    status TEXT NOT NULL DEFAULT 'pending'
);
"""


class SQLiteDatabase:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(SCHEMA)
