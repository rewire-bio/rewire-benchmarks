"""Append-only per-variant log that lets a specialist run resume after interruption.

The first line binds the file to a fingerprint of every setting and artifact
that can change a score. Resuming with any difference is refused, so rows from
two configurations can never be merged. Each row is flushed and fsynced; a
final line cut off by a crash is discarded and rescored. Every loaded row is
validated: a scored row needs a finite score and no reason, an unscored row an
empty score and a reason, and seconds must be finite and nonnegative. An
exclusive lock is held while the log is open, so a second process cannot
append to it. The log holds identifiers, scores and upstream output only, never
labels, and is not a result: result files are written only after every
selected variant has a row.
"""
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path

FIELDS = ("id", "status", "score", "seconds", "reason", "raw")


def fingerprint(settings):
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()


def _clean(value):
    return str(value).replace("\t", " ").replace("\n", " ").replace("\r", " ")


def _finite(text):
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def validate_row(row):
    """Raise ValueError unless a checkpoint row is internally consistent."""
    where = f"checkpoint row {row.get('id')!r}"
    if not row["id"]:
        raise ValueError("checkpoint row without an ID")
    seconds = _finite(row["seconds"])
    if seconds is None or seconds < 0:
        raise ValueError(f"{where}: seconds must be finite and nonnegative")
    if row["status"] == "scored":
        if _finite(row["score"]) is None:
            raise ValueError(f"{where}: scored rows need a finite score")
        if row["reason"]:
            raise ValueError(f"{where}: scored rows carry no unscored reason")
    elif row["status"] == "unscored":
        if row["score"] != "":
            raise ValueError(f"{where}: unscored rows carry no score")
        if not row["reason"]:
            raise ValueError(f"{where}: unscored rows need a reason")
    else:
        raise ValueError(f"{where}: invalid status {row['status']!r}")


class Checkpoint:
    def __init__(self, path, settings):
        self.path = Path(path)
        self.fingerprint = fingerprint(settings)
        self.rows = {}
        self.discarded_partial_line = False
        self.resumed = self.path.exists()
        if not self.resumed:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("x") as handle:
                handle.write(f"#fingerprint\t{self.fingerprint}\t"
                             f"{json.dumps(settings, sort_keys=True)}\n")
                handle.write("\t".join(FIELDS) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        self._handle = self.path.open("r+b")
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._handle.close()
            raise RuntimeError(f"{self.path} is in use by another process") from None
        try:
            self._load()
        except Exception:
            self.close()
            raise

    def close(self):
        if self._handle and not self._handle.closed:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            self._handle.close()

    def _load(self):
        data = self._handle.read()
        if not data.endswith(b"\n"):
            data = data[:data.rfind(b"\n") + 1]
            self.discarded_partial_line = True
            self._handle.truncate(len(data))
            os.fsync(self._handle.fileno())
        lines = data.decode().splitlines()
        if len(lines) < 2 or not lines[0].startswith("#fingerprint\t"):
            raise ValueError(f"{self.path} is not a run checkpoint")
        if lines[0].split("\t")[1] != self.fingerprint:
            raise ValueError(f"{self.path} was written with different settings or artifacts; "
                             "refusing to resume into it")
        if tuple(lines[1].split("\t")) != FIELDS:
            raise ValueError(f"{self.path} has an unexpected header")
        for line in lines[2:]:
            values = line.split("\t")
            if len(values) != len(FIELDS):
                raise ValueError(f"{self.path} has a malformed row")
            row = dict(zip(FIELDS, values))
            if row["id"] in self.rows:
                raise ValueError(f"{self.path} has a duplicate row for {row['id']}")
            validate_row(row)
            self.rows[row["id"]] = row

    def get(self, variant_id):
        return self.rows.get(variant_id)

    def append(self, variant_id, score, seconds, reason="", raw=""):
        if self._handle.closed:
            raise ValueError("checkpoint is closed")
        if variant_id in self.rows:
            raise ValueError(f"duplicate checkpoint row for {variant_id}")
        row = {"id": variant_id, "status": "unscored" if score is None else "scored",
               "score": "" if score is None else repr(float(score)),
               "seconds": f"{seconds:.6f}", "reason": _clean(reason), "raw": _clean(raw)}
        validate_row(row)
        self._handle.seek(0, os.SEEK_END)
        self._handle.write(("\t".join(row[f] for f in FIELDS) + "\n").encode())
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self.rows[variant_id] = row
        return row

    def check_complete(self, variant_ids):
        extra = set(self.rows) - set(variant_ids)
        missing = [v for v in variant_ids if v not in self.rows]
        if extra or missing:
            raise ValueError(f"Checkpoint incomplete or foreign: {len(missing)} missing, "
                             f"{len(extra)} unexpected rows")
