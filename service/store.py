"""SQLite store: the single source of truth for releases, runs, model versions and published nowcasts.

Exactly-once publishing rests on unique constraints, not on workers behaving:
  releases(target, ref_year)   a release is recorded once however often it is detected
  runs(release_id)             one run per release
  model_versions(created_from_run)  the gate decision is durable once made
  nowcasts(run_id)             one nowcast (and one brief) per run
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SCHEMA = """
CREATE TABLE IF NOT EXISTS releases (
    id INTEGER PRIMARY KEY,
    target TEXT NOT NULL,
    ref_year INTEGER NOT NULL,
    ready_at TEXT NOT NULL,
    source TEXT NOT NULL,
    UNIQUE (target, ref_year)
);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY,
    release_id INTEGER NOT NULL UNIQUE REFERENCES releases(id),
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending','fitting','validating','publishing','done','dead_letter')),
    worker_id TEXT,
    lease_until REAL,
    attempt INTEGER NOT NULL DEFAULT 0,
    next_attempt_at REAL NOT NULL DEFAULT 0,
    error TEXT,
    created_at REAL NOT NULL,
    started_at REAL,
    finished_at REAL
);
CREATE TABLE IF NOT EXISTS model_versions (
    id INTEGER PRIMARY KEY,
    created_from_run INTEGER NOT NULL UNIQUE REFERENCES runs(id),
    target TEXT NOT NULL,
    spec TEXT NOT NULL,
    rmspe_candidate REAL,
    rmspe_incumbent REAL,
    rmspe_naive REAL,
    gate_years TEXT NOT NULL,
    decision TEXT NOT NULL CHECK (decision IN ('promoted','rejected','bootstrap','retained')),
    reason TEXT NOT NULL,
    is_incumbent INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS one_incumbent_per_target ON model_versions(target) WHERE is_incumbent = 1;
CREATE TABLE IF NOT EXISTS nowcasts (
    id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL UNIQUE REFERENCES runs(id),
    model_version_id INTEGER NOT NULL REFERENCES model_versions(id),
    produced_by TEXT NOT NULL,
    target TEXT NOT NULL,
    data_through INTEGER NOT NULL,
    forecast_year INTEGER NOT NULL,
    published_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS nowcast_values (
    nowcast_id INTEGER NOT NULL REFERENCES nowcasts(id),
    grp TEXT NOT NULL,
    last_actual REAL NOT NULL,
    value REAL NOT NULL,
    lo REAL,
    hi REAL,
    PRIMARY KEY (nowcast_id, grp)
);
CREATE TABLE IF NOT EXISTS briefs (
    id INTEGER PRIMARY KEY,
    nowcast_id INTEGER NOT NULL UNIQUE REFERENCES nowcasts(id),
    text TEXT NOT NULL,
    passed INTEGER NOT NULL,
    numbers TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY,
    run_id INTEGER,
    worker_id TEXT,
    attempt INTEGER,
    kind TEXT NOT NULL,
    detail TEXT,
    at REAL NOT NULL
);
"""


def db_path() -> Path:
    return Path(os.environ.get("SERVICE_DB", ROOT / "data" / "service.db"))


def connect(path: str | Path | None = None) -> sqlite3.Connection:
    p = Path(path or db_path())
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p, timeout=30, isolation_level=None)  # explicit transactions below
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.executescript(SCHEMA)
    return conn


@contextmanager
def tx(conn: sqlite3.Connection):
    """A write transaction that takes the write lock up front (BEGIN IMMEDIATE), so claims never race."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")


def log_event(conn, run_id, worker_id, attempt, kind, detail=None) -> None:
    conn.execute("INSERT INTO events (run_id, worker_id, attempt, kind, detail, at) VALUES (?,?,?,?,?,?)",
                 (run_id, worker_id, attempt, kind, json.dumps(detail, default=str) if detail is not None else None, time.time()))


def record_release(conn, target: str, ref_year: int, ready_at, source: str) -> bool:
    """Insert one release and its run. Returns True if the release was new; detecting it again inserts nothing."""
    with tx(conn):
        cur = conn.execute("INSERT OR IGNORE INTO releases (target, ref_year, ready_at, source) VALUES (?,?,?,?)",
                           (target, int(ref_year), str(ready_at), source))
        if cur.rowcount == 0:
            return False
        conn.execute("INSERT OR IGNORE INTO runs (release_id, status, created_at) VALUES (?, 'pending', ?)",
                     (cur.lastrowid, time.time()))
        return True
