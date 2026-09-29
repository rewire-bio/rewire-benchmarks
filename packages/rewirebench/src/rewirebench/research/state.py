"""SQLite checkpoints with one live supervisor and immutable completed attempts."""
from __future__ import annotations

import json
import os
import signal
import sqlite3
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import psutil

from .contracts import canonical


def now():
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class Store:
    def __init__(self, workspace, *, create=False):
        self.workspace = Path(workspace).resolve()
        self.path = self.workspace/"campaign.sqlite3"
        if not create and not self.path.is_file():
            raise ValueError("No existing research campaign at this workspace")
        self.db = sqlite3.connect(self.path, timeout=10)
        self.db.row_factory = sqlite3.Row
        if not create:
            try:
                row = self.db.execute("SELECT value FROM meta WHERE key='id'").fetchone()
                if not row or not json.loads(row[0]).startswith("campaign-"):
                    raise ValueError("Invalid campaign identity")
            except (sqlite3.DatabaseError, TypeError, ValueError) as exc:
                self.db.close()
                raise ValueError("Not a valid research campaign database") from exc
            return
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS cases (id TEXT PRIMARY KEY, manifest TEXT NOT NULL,
            state TEXT NOT NULL DEFAULT 'pending', round INTEGER NOT NULL DEFAULT 0,
            error TEXT, bundle TEXT);
        CREATE TABLE IF NOT EXISTS plans (id TEXT PRIMARY KEY, case_id TEXT NOT NULL,
            round INTEGER NOT NULL, spec TEXT NOT NULL, sha256 TEXT NOT NULL,
            UNIQUE(case_id,round));
        CREATE TABLE IF NOT EXISTS attempts (id TEXT PRIMARY KEY, case_id TEXT NOT NULL,
            fingerprint TEXT UNIQUE NOT NULL, record TEXT NOT NULL, state TEXT NOT NULL,
            pid INTEGER, pid_created REAL);
        CREATE TABLE IF NOT EXISTS reasoning (id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id TEXT NOT NULL, role TEXT NOT NULL, round INTEGER NOT NULL,
            state TEXT NOT NULL, payload TEXT, metadata TEXT, error TEXT,
            pid INTEGER, pid_created REAL,
            UNIQUE(case_id,role,round));
        """)

    def close(self):
        self.db.close()

    def get(self, key, default=None):
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def put(self, key, value):
        with self.db:
            self.db.execute("INSERT INTO meta VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                            (key, canonical(value)))

    def status(self):
        cases = [dict(r) for r in self.db.execute("SELECT id,state,round,error FROM cases ORDER BY id")]
        attempts = [dict(r) for r in self.db.execute("SELECT state,COUNT(*) AS count FROM attempts GROUP BY state")]
        return {"campaign_id": self.get("id"), "state": self.get("state"),
                "created_at": self.get("created_at"), "deadline": self.get("deadline"),
                "codex_calls_used": self.get("codex_calls_used", 0), "limits": self.get("limits"),
                "stop_requested": self.get("stop_requested", False), "cases": cases,
                "attempts": attempts, "error": self.get("error")}

    @contextmanager
    def lease(self):
        self.db.execute("BEGIN IMMEDIATE")
        owner = self.get("owner")
        if owner:
            try:
                previous = psutil.Process(owner["pid"])
                if abs(previous.create_time()-owner["created"]) < .01 and previous.is_running():
                    raise ValueError("A supervisor is already running for this campaign")
            except psutil.NoSuchProcess:
                pass
        me = {"pid": os.getpid(), "created": psutil.Process().create_time(), "since": time.time()}
        self.db.execute("INSERT INTO meta VALUES ('owner',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                        (canonical(me),))
        self.db.commit()
        try:
            self.recover()
            yield
        finally:
            self.put("owner", None)

    def recover(self):
        # Interrupted operations are retained and never automatically rerun.
        for row in self.db.execute("SELECT * FROM attempts WHERE state='running'").fetchall():
            _recover_process(row["pid"], row["pid_created"])
            record = json.loads(row["record"])
            record.update(status="interrupted", finished_at=now(), error="Supervisor interrupted; attempt retained")
            with self.db:
                self.db.execute("UPDATE attempts SET record=?,state='interrupted' WHERE id=?",
                                (canonical(record), row["id"]))
        for row in self.db.execute("SELECT pid,pid_created FROM reasoning WHERE state='running'").fetchall():
            _recover_process(row["pid"], row["pid_created"])
        with self.db:
            self.db.execute("UPDATE reasoning SET state='interrupted',error='Session interrupted; budget remains consumed' WHERE state='running'")

    def case(self, identifier):
        row = self.db.execute("SELECT * FROM cases WHERE id=?", (identifier,)).fetchone()
        if not row:
            raise ValueError("Unknown investigation case")
        return dict(row)

    def set_case(self, identifier, state, *, error=None, bundle=None):
        with self.db:
            self.db.execute("UPDATE cases SET state=?,error=?,bundle=COALESCE(?,bundle) WHERE id=?",
                            (state, error, canonical(bundle) if bundle else None, identifier))


def _recover_process(pid, created):
    if not pid:
        return
    try:
        process = psutil.Process(pid)
        if abs(process.create_time()-(created or 0)) >= .01:
            return
        children = process.children(recursive=True)
    except psutil.Error:
        return
    for child in children:
        try:
            child.kill()
        except psutil.Error:
            pass
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        process.kill()
        process.wait(timeout=2)
    except psutil.Error:
        pass
