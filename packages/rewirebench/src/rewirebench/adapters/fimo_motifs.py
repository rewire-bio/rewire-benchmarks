"""Total FIMO motif hits per sequence: a Rewire DART Task 1 reference aggregation.

Registered method identity: ``dart-h12core-fimo-hit-count-v1``
(``H12CoreFimoHitCount``). ``FimoHitCount`` is the same scan and count for
caller-supplied motifs, under its own identity. Scan each supplied
sequence, on both strands, with the complete HOCOMOCO v12 H12CORE collection
(1,443 motifs) using FIMO from MEME Suite 5.5.9, then count every emitted hit.
Every hit has weight 1; zero hits scores 0. Higher means more recognised motif
occurrences. This is a newly specified scalar aggregation using a sourced scanner,
not a published DART Task 1 baseline, and it has no measured performance here.

Fixed FIMO configuration: ``--text --verbosity 1 --thresh 1e-4 --motif-pseudo 0.1
--bfile <uniform A=C=G=T=0.25 file>``. Text mode prints hits as they are found,
so FIMO's stored-score cap (``--max-stored-scores``) never drops hits. FIMO
reports a window when its rounded p-value is <= the threshold, and skips windows
containing a non-ACGT symbol. A sequence with no all-ACGT window as long as the
narrowest motif is unscored rather than given a zero count.

Only ``id`` and ``sequence`` are read. FASTA names are batch positions, so IDs
never reach FIMO. The motifs were learned from external experimental
TF-binding data (ChIP-seq, HT-SELEX and related assays); overlap with ENCODE
regions used by DART is not quantified.
"""
from __future__ import annotations

import csv
import hashlib
import os
import re
import subprocess
import tempfile
from importlib.resources import files
from pathlib import Path

BASELINE_ID = "dart-h12core-fimo-hit-count-v1"
GENERIC_METHOD_ID = "fimo-total-hit-count-caller-motifs-v1"
MEME_VERSION = "5.5.9"
H12CORE_SHA256 = "3d9c47dc396b3ba4e278cdfd29d526c1eb2fbad4895b5e2dc345f535f98c5f12"
H12CORE_MOTIF_COUNT = 1443
H12CORE_IDS_SHA256 = "7bff4fc55e462150c6694d5a13b3cdf7ad9bae81e78113d18013dddb2d2fc308"
H12CORE_URL = ("https://hocomoco12.autosome.org/final_bundle/hocomoco12/H12CORE/"
               "formatted_motifs/H12CORE_meme_format.meme")
BACKGROUND_SHA256 = "b496ad5c41c52729451308eede91b375d03ab4f465394a043fc30ffa3e26e223"
THRESHOLD = "1e-4"
MOTIF_PSEUDOCOUNT = "0.1"
TEXT_MODE_WARNING = "Warning: text mode turns off computation of q-values"
HEADER = ["motif_id", "motif_alt_id", "sequence_name", "start", "stop", "strand",
          "score", "p-value", "q-value", "matched_sequence"]
UNSCORED_NO_WINDOW = "no_all_ACGT_window_as_long_as_narrowest_motif"
TRAINING_OVERLAP = (
    "No DART labels, pairs or controls. External prior: HOCOMOCO v12 motifs learned from "
    "experimental TF-binding data (ChIP-seq, HT-SELEX, methyl-HT-SELEX, SMiLE-seq); overlap "
    "with ENCODE regions used by DART is not quantified. Related TF subtypes are counted "
    "separately and may be overweighted."
)


def _sha(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def background_path() -> Path:
    return Path(str(files("rewirebench").joinpath("resources", "dart_eval", "uniform-background.txt")))


def read_meme_motifs(source: Path | bytes) -> list[tuple[str, int]]:
    """Return (motif ID, width) in file order after minimal MEME format checks."""
    try:
        text = (source if isinstance(source, bytes) else Path(source).read_bytes()).decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("Motif file is not ASCII MEME format") from exc
    if not text.startswith("MEME version"):
        raise ValueError("Motif file is not MEME format")
    if not re.search(r"^ALPHABET= ACGT\s*$", text, re.MULTILINE):
        raise ValueError("Motif file must declare the DNA ACGT alphabet")
    strands = re.search(r"^strands:(.*)$", text, re.MULTILINE)
    if strands and strands.group(1).split() != ["+", "-"]:
        raise ValueError("Motif file must allow scanning both strands")
    ids = re.findall(r"^MOTIF\s+(\S+)", text, re.MULTILINE)
    widths = [int(w) for w in re.findall(r"^letter-probability matrix:.*\bw=\s*(\d+)", text, re.MULTILINE)]
    if not ids or len(ids) != len(widths) or len(set(ids)) != len(ids) or min(widths) < 1:
        raise ValueError("Motif file needs unique motif IDs, each with one probability matrix")
    return list(zip(ids, widths))


def motif_ids_digest(motifs: list[tuple[str, int]]) -> str:
    return hashlib.sha256("".join(f"{ident}\n" for ident, _ in motifs).encode()).hexdigest()


def _longest_acgt_run(sequence: str) -> int:
    return max((len(run) for run in re.split(r"[^ACGT]", sequence)), default=0)


def _valid_windows(sequence: str, width: int) -> int:
    return sum(len(run) - width + 1 for run in re.split(r"[^ACGT]", sequence) if len(run) >= width)


class FimoHitCount:
    """Total FIMO hits for caller-supplied motifs: ``fimo-total-hit-count-caller-motifs-v1``.

    A generic scanner, not the registered DART reference. Its provenance names
    the caller's motif hash and their declared ``motif_provenance``; it never
    claims HOCOMOCO training data. ``H12CoreFimoHitCount`` is the registered
    ``dart-h12core-fimo-hit-count-v1`` configuration.

    ``fimo`` is a filesystem path, resolved against the current directory (never
    searched on PATH) before hashing, and that resolved file is what runs.
    ``fimo_sha256`` is the caller's recorded digest of their FIMO build; it must
    match, and ``fimo --version`` must be 5.5.9. Motif and background bytes are
    verified once and scanned from private copies. Before every FIMO call the
    executable, motif file and background file are hashed again, and a changed
    file stops the scan.
    """

    INPUT_FIELDS = frozenset({"id", "sequence"})

    def __init__(self, fimo: str | Path, motifs: str | Path, *, fimo_sha256: str,
                 motifs_sha256: str, motif_provenance: str, motif_count: int | None = None,
                 motif_ids_sha256: str | None = None, timeout: float = 3600):
        if not isinstance(motif_provenance, str) or not motif_provenance.strip():
            raise ValueError("motif_provenance must describe the supplied motifs")
        try:
            self.fimo = Path(fimo).resolve(strict=True)
            self.motifs = Path(motifs).resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ValueError("FIMO executable or motif file is missing") from exc
        if not self.fimo.is_file() or not os.access(self.fimo, os.X_OK):
            raise ValueError("FIMO executable is missing or not executable")
        self.fimo_sha256 = _sha(self.fimo)
        if self.fimo_sha256 != fimo_sha256:
            raise ValueError("FIMO executable does not match fimo_sha256")
        version = subprocess.run([str(self.fimo), "--version"], capture_output=True, text=True,
                                 timeout=60, check=False)
        if version.returncode != 0 or version.stdout.strip() != MEME_VERSION:
            raise ValueError(f"FIMO must report version {MEME_VERSION}")
        self._motif_bytes = self.motifs.read_bytes() if self.motifs.is_file() else b""
        if hashlib.sha256(self._motif_bytes).hexdigest() != motifs_sha256:
            raise ValueError("Motif file does not match motifs_sha256")
        motif_list = read_meme_motifs(self._motif_bytes)
        if motif_count is not None and len(motif_list) != motif_count:
            raise ValueError(f"Motif file must contain {motif_count} motifs")
        if motif_ids_sha256 is not None and motif_ids_digest(motif_list) != motif_ids_sha256:
            raise ValueError("Motif IDs do not match the pinned collection")
        self.background = background_path().resolve(strict=True)
        self._background_bytes = self.background.read_bytes()
        if hashlib.sha256(self._background_bytes).hexdigest() != BACKGROUND_SHA256:
            raise ValueError("Packaged uniform background does not match its pin")
        self.motifs_sha256 = motifs_sha256
        self.widths = sorted({w for _, w in motif_list})
        self.timeout = timeout
        self.arguments = ["--text", "--verbosity", "1", "--thresh", THRESHOLD,
                          "--motif-pseudo", MOTIF_PSEUDOCOUNT, "--bfile"]
        self._coverage = {"sequences": 0, "with_non_ACGT": 0, "unscored_no_valid_window": 0,
                          "positions": 0, "valid_windows_narrowest_motif": 0,
                          "valid_windows_widest_motif": 0, "hits": 0, "zero_hit_sequences": 0}
        self._identity = {
            "fimo_version": MEME_VERSION, "fimo_sha256": self.fimo_sha256,
            "motifs_sha256": motifs_sha256, "motif_count": len(motif_list),
            "motif_ids_sha256": motif_ids_digest(motif_list),
            "background_sha256": BACKGROUND_SHA256, "fimo_arguments": self.arguments + ["<background>"],
            "implementation_sha256": _sha(Path(__file__)),
            "aggregation": "total emitted hits over all motifs and both strands; weight 1",
            "published_result_reproduction": False,
        }
        self._set_identity(
            method_id=GENERIC_METHOD_ID, motif_collection="caller-supplied",
            motif_provenance=motif_provenance,
            training_overlap=("No DART labels, pairs or controls. Caller-supplied motifs; their "
                              "training data and overlap are as declared by the caller and not checked: "
                              + motif_provenance))

    def _set_identity(self, **fields):
        for key in ("baseline_id", "method_id", "motif_collection", "motif_provenance", "training_overlap"):
            self._identity.pop(key, None)
        self._identity.update(fields)
        configuration = "\n".join([fields["method_id"], self.fimo_sha256, self.motifs_sha256,
                                   BACKGROUND_SHA256, *self.arguments])
        self._identity["configuration_sha256"] = hashlib.sha256(configuration.encode()).hexdigest()

    @property
    def provenance(self) -> dict:
        return {**self._identity, "scan_coverage": dict(self._coverage)}

    def _validate(self, inputs):
        seen = set()
        for row in inputs:
            if set(row) != self.INPUT_FIELDS:
                raise ValueError("FIMO adapter accepts only id and sequence")
            sequence = row["sequence"]
            if not isinstance(sequence, str) or not sequence or set(sequence) - set("ACGTN"):
                raise ValueError("Sequences must be nonempty uppercase ACGTN text")
            if row["id"] in seen:
                raise ValueError("Duplicate input ID")
            seen.add(row["id"])

    def _check_unchanged(self):
        for path, digest in ((self.fimo, self.fimo_sha256), (self.motifs, self.motifs_sha256),
                             (self.background, BACKGROUND_SHA256)):
            try:
                current = _sha(path)
            except OSError:
                current = None
            if current != digest:
                raise RuntimeError(f"{path.name} changed after the adapter was constructed; scan stopped")

    def hits(self, inputs) -> dict:
        """Emitted FIMO hits per scorable input ID; unscorable IDs map to None."""
        self._validate(inputs)
        result, names = {}, {}
        for row in inputs:
            sequence = row["sequence"]
            self._coverage["sequences"] += 1
            self._coverage["positions"] += len(sequence)
            self._coverage["with_non_ACGT"] += bool(set(sequence) - set("ACGT"))
            self._coverage["valid_windows_narrowest_motif"] += _valid_windows(sequence, self.widths[0])
            self._coverage["valid_windows_widest_motif"] += _valid_windows(sequence, self.widths[-1])
            if _longest_acgt_run(sequence) < self.widths[0]:
                self._coverage["unscored_no_valid_window"] += 1
                result[row["id"]] = None
                continue
            names[f"s{len(names)}"] = row
            result[row["id"]] = []
        if not names:
            return result
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "motifs.meme").write_bytes(self._motif_bytes)
            (tmp / "background.txt").write_bytes(self._background_bytes)
            fasta = tmp / "input.fa"
            fasta.write_text("".join(f">{name}\n{row['sequence']}\n" for name, row in names.items()))
            self._check_unchanged()
            run = subprocess.run(
                [str(self.fimo), *self.arguments, str(tmp / "background.txt"),
                 str(tmp / "motifs.meme"), str(fasta)],
                capture_output=True, text=True, timeout=self.timeout, check=False)
        if run.returncode != 0:
            raise RuntimeError(f"FIMO exited with status {run.returncode}")
        if [line for line in run.stderr.splitlines() if line.strip()] != [TEXT_MODE_WARNING]:
            raise RuntimeError("FIMO emitted unexpected diagnostics; hits may be incomplete")
        lines = run.stdout.splitlines()
        if lines:
            if lines[0].split("\t") != HEADER:
                raise RuntimeError("Unexpected FIMO text header")
            for record in csv.reader(lines[1:], delimiter="\t"):
                if len(record) != len(HEADER) or record[2] not in names:
                    raise RuntimeError("Unparseable FIMO hit or unknown sequence name")
                row = names[record[2]]
                try:
                    start, stop = int(record[3]), int(record[4])
                except ValueError as exc:
                    raise RuntimeError("Unparseable FIMO hit coordinates") from exc
                if record[5] not in ("+", "-") or not 1 <= min(start, stop) <= max(start, stop) <= len(row["sequence"]):
                    raise RuntimeError("FIMO hit outside the scanned sequence")
                result[row["id"]].append((record[0], start, stop, record[5], record[6], record[7]))
        return result

    def predict(self, inputs):
        results = {}
        for ident, found in self.hits(inputs).items():
            if found is None:
                results[ident] = {"score": None, "reason": UNSCORED_NO_WINDOW}
                continue
            results[ident] = len(found)
            self._coverage["hits"] += len(found)
            self._coverage["zero_hit_sequences"] += not found
        return results


class H12CoreFimoHitCount(FimoHitCount):
    """Registered ``dart-h12core-fimo-hit-count-v1``: the exact 1,443-motif H12CORE file.

    The H12CORE identity and HOCOMOCO training statement are set only after the
    file digest, motif count and ordered motif-ID digest have been checked.
    """

    def __init__(self, fimo: str | Path, motifs: str | Path, *, fimo_sha256: str,
                 timeout: float = 3600):
        super().__init__(fimo, motifs, fimo_sha256=fimo_sha256, motifs_sha256=H12CORE_SHA256,
                         motif_provenance=f"HOCOMOCO v12 H12CORE MEME file, {H12CORE_URL}",
                         motif_count=H12CORE_MOTIF_COUNT, motif_ids_sha256=H12CORE_IDS_SHA256,
                         timeout=timeout)
        self._set_identity(
            baseline_id=BASELINE_ID, method_id=BASELINE_ID, motif_collection="HOCOMOCO v12 H12CORE",
            motif_provenance=f"HOCOMOCO v12 H12CORE MEME file, {H12CORE_URL}",
            training_overlap=TRAINING_OVERLAP)
