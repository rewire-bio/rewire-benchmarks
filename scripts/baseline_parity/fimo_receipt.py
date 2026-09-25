"""Record hits from a FIMO 5.5.9 binary on the synthetic DART motif fixture.

Independent of rewirebench: this invokes FIMO directly with the reviewed
configuration and parses its text output itself. The receipt records every
emitted (motif, sequence, start, stop, strand, score, p-value) tuple, the binary
and motif hashes, and two synthetic scanner checks:

- threshold equality: a single sharp width-8 site whose p-value FIMO rounds to
  exactly 1.525878906e-05 (4^-8 to 10 significant digits), scanned at that
  threshold and at 1.525878905e-05;
- stored-score cap: the 350-base poly-A sequence scanned with
  ``--max-stored-scores 50`` in stored mode and in text mode.

Build provenance is recorded only from a build receipt whose ``fimo_sha256``
matches the executable actually run (``--build-receipt``). Without one, the
receipt says so and lists reference build instructions, not observed provenance.

    python scripts/baseline_parity/fimo_receipt.py --fimo /path/to/fimo \
        --h12core /path/to/H12CORE_meme_format.meme [--build-receipt build.json] \
        --output receipt.json
"""
import argparse
import hashlib
import json
import platform
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "packages/rewirebench/tests/fixtures/dart_motif"
BACKGROUND = ROOT / "packages/rewirebench/src/rewirebench/resources/dart_eval/uniform-background.txt"
CONFIGURATION = ["--text", "--verbosity", "1", "--thresh", "1e-4", "--motif-pseudo", "0.1"]
REFERENCE_BUILD = {
    "status": "reference_instructions_not_observed_provenance",
    "source": "https://meme-suite.org/meme/meme-software/5.5.9/meme-5.5.9.tar.gz "
              "(sha256 0406fb7b1dc27f6aab3d6d3a29ecdf617bbdd946690cfa97ca2129bc210bfd12)",
    "commands": ["./configure --prefix=<prefix> --enable-build-libxml2 --enable-build-libxslt",
                 "make", "make install"],
}
EQUAL_THRESHOLD, BELOW_THRESHOLD = "1.525878906e-05", "1.525878905e-05"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_fimo(fimo, options, motifs, sequences, *, text=True):
    with tempfile.TemporaryDirectory() as tmp:
        fasta = Path(tmp) / "fixture.fa"
        fasta.write_text("".join(f">{name}\n{seq}\n" for name, seq in sequences.items()))
        if text:
            run = subprocess.run([fimo, *options, "--bfile", str(BACKGROUND), str(motifs), str(fasta)],
                                 capture_output=True, text=True, check=True)
            lines = run.stdout.splitlines()
        else:
            out = Path(tmp) / "out"
            run = subprocess.run([fimo, "--oc", str(out), *options, "--bfile", str(BACKGROUND),
                                  str(motifs), str(fasta)], capture_output=True, text=True, check=True)
            lines = (out / "fimo.tsv").read_text().splitlines()
    rows = [line.split("\t") for line in lines
            if line and not line.startswith("#") and not line.startswith("motif_id")]
    hits = {name: [] for name in sequences}
    for row in rows:
        hits[row[2]].append([row[0], int(row[3]), int(row[4]), row[5], row[6], row[7]])
    return hits, run.stderr.splitlines()


def build_provenance(path, fimo_digest):
    if path is None:
        return {**REFERENCE_BUILD, "note": "No build receipt supplied for this executable"}
    receipt = json.loads(Path(path).read_text())
    if receipt.get("fimo_sha256") != fimo_digest:
        raise SystemExit("Build receipt describes a different FIMO executable")
    return {"status": "build_receipt_bound_to_executable_digest",
            "build_receipt_sha256": sha(path), "build_receipt": receipt}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fimo", required=True, type=Path)
    parser.add_argument("--h12core", required=True, type=Path)
    parser.add_argument("--build-receipt", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    fimo = str(args.fimo.resolve(strict=True))  # hash and run the same file, never a PATH lookup
    fimo_digest = sha(fimo)
    provenance = build_provenance(args.build_receipt, fimo_digest)
    version = subprocess.run([fimo, "--version"], capture_output=True, text=True,
                             check=True).stdout.strip()
    if version != "5.5.9":
        raise SystemExit(f"FIMO version {version}, expected 5.5.9")
    sequences = json.loads((FIXTURE / "sequences.json").read_text())
    motifs = FIXTURE / "synthetic.meme"
    synthetic, stderr = run_fimo(fimo, CONFIGURATION, motifs, sequences)
    randoms = {k: v for k, v in sequences.items() if k.startswith("random_")}
    h12core, h12_stderr = run_fimo(fimo, CONFIGURATION, args.h12core.resolve(strict=True), randoms)
    high = {"high_hit": sequences["high_hit"]}
    capped = ["--verbosity", "1", "--thresh", "1e-4", "--motif-pseudo", "0.1", "--max-stored-scores", "50"]
    stored, _ = run_fimo(fimo, capped, motifs, high, text=False)
    text_capped, _ = run_fimo(fimo, ["--text", *capped], motifs, high)
    site = {"single_polya_site": "C" * 100 + "A" * 8 + "C" * 100}
    boundary = {}
    for label, threshold in (("equal", EQUAL_THRESHOLD), ("below", BELOW_THRESHOLD)):
        options = ["--text", "--verbosity", "1", "--thresh", threshold, "--motif-pseudo", "0.1",
                   "--motif", "SYN_POLYA"]
        found, _ = run_fimo(fimo, options, motifs, site)
        boundary[label] = {"thresh": threshold, "hits": found["single_polya_site"]}
    receipt = {
        "schema": "rewire-fimo-receipt-v2", "created_at": datetime.now(UTC).isoformat(),
        "fimo_version": version, "fimo_sha256": fimo_digest, "build_provenance": provenance,
        "platform": platform.platform(), "configuration": CONFIGURATION + ["--bfile", "<background>"],
        "background_sha256": sha(BACKGROUND), "synthetic_meme_sha256": sha(motifs),
        "sequences_sha256": sha(FIXTURE / "sequences.json"),
        "h12core_sha256": sha(args.h12core),
        "stderr": sorted(set(stderr + h12_stderr)),
        "synthetic_hits": synthetic, "h12core_hits": h12core,
        "cap_50_high_hit_rows": {"stored_mode": len(stored["high_hit"]),
                                 "text_mode": len(text_capped["high_hit"])},
        "threshold_boundary": {
            "motif": "SYN_POLYA", "sequence": "100 C, 8 A, 100 C",
            "expected_internal_p_value": "4^-8 = 1.52587890625e-05, rounded by FIMO to 10 significant "
                                         "digits = 1.525878906e-05, the same double as the equal threshold",
            **boundary,
        },
    }
    args.output.write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
