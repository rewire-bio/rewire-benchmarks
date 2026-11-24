"""Registered score-table lookup adapter for ClinGen PP3/BP4 calibration."""

from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path
from typing import Any


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ScoreTableLookup:
    """Look up precomputed per-variant scores from a SHA-256-verified TSV."""

    def __init__(self, artifact: str, artifact_sha256: str, column: str = "score"):
        path = Path(artifact)
        if not path.is_file():
            raise FileNotFoundError(f"Score table artifact not found: {artifact}")
        actual_sha = _sha256_file(path)
        expected_sha = str(artifact_sha256).lower()
        if len(expected_sha) != 64 or actual_sha != expected_sha:
            raise ValueError(
                f"Score table artifact SHA-256 mismatch for {path.name}: "
                f"got {actual_sha}, expected {expected_sha}"
            )
        self.artifact = str(path)
        self.artifact_sha256 = actual_sha
        self.column = str(column)
        self._scores: dict[tuple[str, int, str, str], float] = {}
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            fields = set(reader.fieldnames or ())
            required = {"chrom", "pos", "ref", "alt", self.column}
            if not required.issubset(fields):
                missing = ", ".join(sorted(required - fields))
                raise ValueError(f"Score table {path.name} missing required column(s): {missing}")
            for row in reader:
                raw_val = row.get(self.column, "")
                if raw_val in ("", "NA", "nan", "NaN", "None"):
                    continue
                val = float(raw_val)
                if not math.isfinite(val):
                    continue
                self._scores[(row["chrom"], int(row["pos"]), row["ref"], row["alt"])] = val
        self.provenance = {"implementation_sha256": actual_sha}

    def predict(self, inputs: list[dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for row in inputs:
            key = (row["chrom"], int(row["pos"]), row["ref"], row["alt"])
            if key in self._scores:
                out[row["id"]] = self._scores[key]
            else:
                out[row["id"]] = {"score": None, "reason": "variant_not_in_score_table"}
        return out
