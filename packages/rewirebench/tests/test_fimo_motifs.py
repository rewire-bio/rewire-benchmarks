"""H12CORE/FIMO total hit count: failure modes, and parity with a pinned FIMO binary.

Tests marked ``needs_fimo`` run only when REWIRE_FIMO points at a FIMO 5.5.9
executable (and REWIRE_H12CORE at H12CORE_meme_format.meme for the collection
tests). Skipped tests are not parity evidence. fimo-receipt.json records an
independent run of the pinned binary (scripts/baseline_parity/fimo_receipt.py).
Scripted stand-in executables below test failure handling only.
"""
import copy
import csv
import hashlib
import io
import json
import math
import os
import stat
import subprocess
from pathlib import Path

import pytest
from rewirebench import sdk
from rewirebench.adapters import fimo_motifs as fm
from rewirebench.baselines import run_baselines

FIXTURE = Path(__file__).parent / "fixtures" / "dart_motif"
SEQUENCES = json.loads((FIXTURE / "sequences.json").read_text())
RECEIPT = json.loads((FIXTURE / "fimo-receipt.json").read_text())
FIMO = os.environ.get("REWIRE_FIMO")
SYNTHETIC = "hand-authored synthetic test motifs (make_fixture.py); no biological source"
H12CORE = os.environ.get("REWIRE_H12CORE")
needs_fimo = pytest.mark.skipif(not FIMO, reason="REWIRE_FIMO not set; FIMO parity not executed")
needs_h12core = pytest.mark.skipif(not (FIMO and H12CORE), reason="REWIRE_FIMO/REWIRE_H12CORE not set")
EXPECTED = {  # planted by construction in make_fixture.py
    "forward": [("SYN_FWD", 101, 110, "+")],
    "reverse": [("SYN_FWD", 201, 210, "-")],
    "palindrome": [("SYN_PAL", 51, 60, "+"), ("SYN_PAL", 51, 60, "-")],
    "two_sites": [("SYN_FWD", 21, 30, "+"), ("SYN_FWD", 301, 310, "-")],
    "ambiguous_site": [("SYN_FWD", 201, 210, "+")],  # N inside the site at 101 removes that window
    "none": [],
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inputs(names, ids=None):
    return [{"id": (ids or {}).get(n, n), "sequence": SEQUENCES[n]} for n in names]


def synthetic():
    return fm.FimoHitCount(FIMO, FIXTURE / "synthetic.meme", fimo_sha256=sha(FIMO),
                           motifs_sha256=sha(FIXTURE / "synthetic.meme"), motif_provenance=SYNTHETIC)


def independent_fimo(motifs, sequences, *extra, thresh="1e-4"):
    """A direct FIMO call written out here, not through the adapter."""
    fasta = "".join(f">{name}\n{seq}\n" for name, seq in sequences.items())
    path = Path(os.environ.get("TMPDIR", "/tmp")) / f"rewire-fimo-{os.getpid()}.fa"
    path.write_text(fasta)
    try:
        out = subprocess.run([FIMO, "--text", "--verbosity", "1", "--thresh", thresh, "--motif-pseudo", "0.1",
                              "--bfile", str(fm.background_path()), *extra, str(motifs), str(path)],
                             capture_output=True, text=True, check=True).stdout
    finally:
        path.unlink()
    hits = {name: [] for name in sequences}
    for row in list(csv.reader(io.StringIO(out), delimiter="\t"))[1:]:
        hits[row[2]].append((row[0], int(row[3]), int(row[4]), row[5], row[6], row[7]))
    return hits


def test_receipt_matches_current_fixture_and_pins():
    assert RECEIPT["fimo_version"] == fm.MEME_VERSION
    assert RECEIPT["synthetic_meme_sha256"] == sha(FIXTURE / "synthetic.meme")
    assert RECEIPT["sequences_sha256"] == sha(FIXTURE / "sequences.json")
    assert RECEIPT["background_sha256"] == fm.BACKGROUND_SHA256 == sha(fm.background_path())
    assert RECEIPT["h12core_sha256"] == fm.H12CORE_SHA256
    assert RECEIPT["configuration"] == ["--text", "--verbosity", "1", "--thresh", "1e-4",
                                        "--motif-pseudo", "0.1", "--bfile", "<background>"]
    assert RECEIPT["stderr"] == [fm.TEXT_MODE_WARNING]
    for name, expected in EXPECTED.items():
        assert [tuple(hit[:4]) for hit in RECEIPT["synthetic_hits"][name]] == expected
    assert len(RECEIPT["synthetic_hits"]["high_hit"]) == 343  # every overlapping poly-A window
    # Stored mode with a 50-score cap discarded every tied hit; text mode keeps them.
    # With the same 50-score cap, stored mode dropped every tied hit; text mode kept all 343.
    assert RECEIPT["cap_50_high_hit_rows"] == {"stored_mode": 0, "text_mode": 343}
    assert RECEIPT["synthetic_hits"]["no_valid_window"] == []
    boundary = RECEIPT["threshold_boundary"]
    assert [hit[:4] for hit in boundary["equal"]["hits"]] == [["SYN_POLYA", 101, 108, "+"]]
    assert boundary["below"]["hits"] == []


def test_receipt_build_provenance_is_bound_to_the_executable_digest():
    build = json.loads((FIXTURE / "fimo-build-receipt.json").read_text())
    provenance = RECEIPT["build_provenance"]
    assert provenance["status"] == "build_receipt_bound_to_executable_digest"
    assert provenance["build_receipt"] == build
    assert provenance["build_receipt_sha256"] == sha(FIXTURE / "fimo-build-receipt.json")
    assert build["fimo_sha256"] == RECEIPT["fimo_sha256"] == build["build_tree_fimo_sha256"]
    assert build["source"]["sha256"] == "0406fb7b1dc27f6aab3d6d3a29ecdf617bbdd946690cfa97ca2129bc210bfd12"


def test_threshold_fixture_premise_matches_fimo_rounding():
    """The equal threshold is the double FIMO's RND(p, 10) yields for p = 4^-8 (macros.h)."""
    p = 0.25 ** 8
    scale = 10.0 ** math.ceil(10 - 1 - math.log10(p))
    assert round(scale * p) / scale == float("1.525878906e-05") > float("1.525878905e-05")


def test_motif_file_validation(tmp_path):
    motifs = fm.read_meme_motifs(FIXTURE / "synthetic.meme")
    assert motifs == [("SYN_FWD", 10), ("SYN_PAL", 10), ("SYN_POLYA", 8)]
    text = (FIXTURE / "synthetic.meme").read_text()
    for broken, message in [(text.replace("ALPHABET= ACGT", "ALPHABET= ACGU"), "ACGT"),
                            (text.replace("strands: + -", "strands: +"), "both strands"),
                            (text.replace("SYN_PAL", "SYN_FWD"), "unique"),
                            ("not a motif file", "MEME")]:
        (tmp_path / "bad.meme").write_text(broken)
        with pytest.raises(ValueError, match=message):
            fm.read_meme_motifs(tmp_path / "bad.meme")


def stand_in(tmp_path, body, version="5.5.9", name="fimo"):
    """A scripted executable for failure-path tests; never parity evidence."""
    path = tmp_path / name
    path.write_text(f'#!/bin/sh\nif [ "$1" = "--version" ]; then echo {version}; exit 0; fi\n{body}\n')
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def construct(executable, **kw):
    motifs = kw.pop("motifs", FIXTURE / "synthetic.meme")
    return fm.FimoHitCount(executable, motifs, fimo_sha256=kw.pop("fimo_sha256", sha(executable)),
                           motifs_sha256=kw.pop("motifs_sha256", sha(motifs)),
                           motif_provenance=kw.pop("motif_provenance", SYNTHETIC), **kw)


def test_missing_or_mismatched_prerequisites_block(tmp_path):
    with pytest.raises(ValueError, match="missing"):
        fm.FimoHitCount(tmp_path / "absent", FIXTURE / "synthetic.meme", fimo_sha256="0" * 64,
                        motifs_sha256=sha(FIXTURE / "synthetic.meme"), motif_provenance=SYNTHETIC)
    with pytest.raises(ValueError, match="version 5.5.9"):
        construct(stand_in(tmp_path, "exit 0", version="5.5.8"))
    good = stand_in(tmp_path, "exit 0")
    with pytest.raises(ValueError, match="fimo_sha256"):
        construct(good, fimo_sha256="0" * 64)
    with pytest.raises(ValueError, match="motifs_sha256"):
        construct(good, motifs_sha256="0" * 64)
    with pytest.raises(ValueError, match="3 motifs|1443 motifs"):
        construct(good, motif_count=1443)
    with pytest.raises(ValueError, match="motifs_sha256"):
        fm.H12CoreFimoHitCount(good, FIXTURE / "synthetic.meme", fimo_sha256=sha(good))


@pytest.mark.parametrize("body, error", [
    ("exit 3", "status 3"),
    ('echo "Scaled log-odds score out of range" >&2; exit 0', "unexpected diagnostics"),
    (('echo "Warning: text mode turns off computation of q-values" >&2; '
      'printf "motif_id\\tmotif_alt_id\\tsequence_name\\tstart\\tstop\\tstrand\\tscore\\tp-value\\tq-value\\tmatched_sequence\\n'
      'M\\t.\\tunknown\\t1\\t10\\t+\\t1\\t1e-05\\t\\tA\\n"'), "unknown sequence"),
])
def test_scanner_failures_are_errors_not_zero_scores(tmp_path, body, error):
    adapter = construct(stand_in(tmp_path, body))
    with pytest.raises(RuntimeError, match=error):
        adapter.predict(inputs(["forward"]))


def test_only_sequence_and_id_are_accepted(tmp_path):
    adapter = construct(stand_in(tmp_path, 'echo "Warning: text mode turns off computation of q-values" >&2'))
    for row in ({"id": "a", "sequence": "ACGT" * 10, "target": 1},
                {"id": "a", "sequence": "acgt" * 10}, {"id": "a", "sequence": ""}):
        with pytest.raises(ValueError):
            adapter.predict([row])
    # A sequence without any all-ACGT window as wide as the narrowest motif is unscored.
    assert adapter.predict(inputs(["no_valid_window"])) == {
        "no_valid_window": {"score": None, "reason": fm.UNSCORED_NO_WINDOW}}
    assert not hasattr(adapter, "fit")


@needs_fimo
def test_hits_equal_independent_fimo_run_and_receipt():
    adapter = synthetic()
    assert adapter.fimo_sha256 == RECEIPT["fimo_sha256"], "receipt came from a different binary"
    names = sorted(SEQUENCES)
    found = adapter.hits(inputs(names))
    direct = independent_fimo(FIXTURE / "synthetic.meme", {n: SEQUENCES[n] for n in names if n != "no_valid_window"})
    for name in names:
        if name == "no_valid_window":
            assert found[name] is None
            continue
        assert found[name] == direct[name]
        assert [list(hit) for hit in found[name]] == RECEIPT["synthetic_hits"][name]
    counts = adapter.predict(inputs(names))
    assert counts["high_hit"] == 343 and counts["palindrome"] == 2 and counts["none"] == 0
    for name, expected in EXPECTED.items():
        assert [hit[:4] for hit in found[name]] == expected
        assert counts[name] == len(expected)


@needs_fimo
def test_hits_are_the_threshold_subset_of_a_full_scan():
    """Brackets printed p-values only; equality itself is tested below."""
    everything = independent_fimo(FIXTURE / "synthetic.meme", {"random_a": SEQUENCES["random_a"]}, "--thresh", "1")
    kept = synthetic().hits(inputs(["random_a"]))["random_a"]
    assert len(everything["random_a"]) == 2 * 3 * 341 + 2 * 2  # both strands, all windows (w 10,10,8)
    assert set(kept) <= set(everything["random_a"])
    assert all(float(hit[5]) <= 1e-4 for hit in kept)
    assert all(float(hit[5]) >= 1e-4 for hit in set(everything["random_a"]) - set(kept))


@needs_fimo
def test_threshold_equality_is_inclusive():
    """A site whose rounded p-value equals the threshold is emitted; one step below it is not."""
    site = {"site": "C" * 100 + "A" * 8 + "C" * 100}
    motif = ("--motif", "SYN_POLYA")
    equal = independent_fimo(FIXTURE / "synthetic.meme", site, *motif, thresh="1.525878906e-05")
    below = independent_fimo(FIXTURE / "synthetic.meme", site, *motif, thresh="1.525878905e-05")
    assert [hit[:4] for hit in equal["site"]] == [("SYN_POLYA", 101, 108, "+")]
    assert below["site"] == []


@needs_fimo
def test_text_mode_is_not_truncated_by_a_small_stored_score_cap():
    high = {"high_hit": SEQUENCES["high_hit"]}
    capped = independent_fimo(FIXTURE / "synthetic.meme", high, "--max-stored-scores", "50")
    assert len(capped["high_hit"]) == 343 == len(synthetic().hits(inputs(["high_hit"]))["high_hit"])


@needs_fimo
def test_batch_order_id_and_strand_invariance():
    adapter = synthetic()
    names = sorted(n for n in SEQUENCES if n != "no_valid_window")
    batch = adapter.predict(inputs(names))
    reordered = adapter.predict(inputs(list(reversed(names))))
    singles = {}
    for name in names:
        singles.update(adapter.predict(inputs([name])))
    assert batch == reordered == singles
    renamed = adapter.predict(inputs(names, ids={n: f"opaque-{i}" for i, n in enumerate(names)}))
    assert {f"opaque-{i}": batch[n] for i, n in enumerate(names)} == renamed
    complement = str.maketrans("ACGTN", "TGCAN")
    flipped = adapter.predict([{"id": n, "sequence": SEQUENCES[n].translate(complement)[::-1]} for n in names])
    assert flipped == batch  # uniform background, both strands: reverse complement counts agree
    assert adapter.provenance["scan_coverage"]["with_non_ACGT"] > 0


@needs_h12core
def test_h12core_collection_matches_independent_run_and_rejects_modified_file(tmp_path):
    adapter = fm.H12CoreFimoHitCount(FIMO, H12CORE, fimo_sha256=sha(FIMO))
    names = [n for n in SEQUENCES if n.startswith("random_")]
    found = adapter.hits(inputs(names))
    direct = independent_fimo(H12CORE, {n: SEQUENCES[n] for n in names})
    for name in names:
        assert found[name] == direct[name]
        assert [list(hit) for hit in found[name]] == RECEIPT["h12core_hits"][name]
    assert adapter.provenance["motif_count"] == 1443
    modified = tmp_path / "H12CORE_meme_format.meme"
    modified.write_bytes(Path(H12CORE).read_bytes() + b"\n")
    with pytest.raises(ValueError, match="motifs_sha256"):
        fm.H12CoreFimoHitCount(FIMO, modified, fimo_sha256=sha(FIMO))


def rehash(data):
    data["prepared_sha256"] = sdk._digest({k: v for k, v in data.items() if k != "prepared_sha256"})
    return data


def test_default_dart_batch_blocks_without_scanner(tmp_path):
    data = sdk.prepare("dart-eval-task1-zero-shot-v1", source="demo", output=tmp_path / "prepared")
    manifest = run_baselines(data, output=tmp_path / "run")
    assert [r["status"] for r in manifest["baselines"]] == ["evaluated", "blocked", "blocked"]
    assert "no GC or Markov fallback" in manifest["baselines"][2]["reason"]
    # Equal integer counts are incorrect, not half-correct; ties are reported separately.
    from rewirebench.protocols import dart_eval
    counts = {row["id"]: (2 if row["pair_role"] == "element" and row["source_index"] % 2 else 1)
              for row in data["rows"]}
    result = dart_eval.score(data, counts)
    pairs = result["pair_coverage"]["scored"]
    assert result["pair_diagnostics"]["tied_pairs"] == pairs - round(result["metrics"]["acc"] * pairs)
    assert 0 < result["pair_diagnostics"]["tied_pairs"] < pairs


@needs_h12core
def test_dart_runner_uses_only_sequences(tmp_path):
    data = sdk.prepare("dart-eval-task1-zero-shot-v1", source="demo", output=tmp_path / "prepared")
    options = {fm.BASELINE_ID: {"fimo": FIMO, "fimo_sha256": sha(FIMO), "motifs": H12CORE}}
    run = run_baselines(data, output=tmp_path / "run", baseline_ids=[fm.BASELINE_ID],
                        baseline_options=options, batch_size=5)
    assert run["baselines"][0]["status"] == "evaluated"
    report = json.loads((tmp_path / "run" / fm.BASELINE_ID / "report.json").read_text())
    assert report["execution"]["adapter_provenance"]["motif_count"] == 1443
    assert "HOCOMOCO" in report["model"]["training_overlap"]
    # Swapping pair roles, labels and source indices must not change any score.
    changed = copy.deepcopy(data)
    for row in changed["rows"]:
        row["pair_role"] = "control" if row["pair_role"] == "element" else "element"
        row["target"] = 1 - row["target"]
        row["source_index"] += 1000
    run_baselines(rehash(changed), output=tmp_path / "changed", baseline_ids=[fm.BASELINE_ID],
                  baseline_options=options, batch_size=24)
    assert ((tmp_path / "run" / fm.BASELINE_ID / "predictions.json").read_text()
            == (tmp_path / "changed" / fm.BASELINE_ID / "predictions.json").read_text())


QUIET = 'echo "Warning: text mode turns off computation of q-values" >&2'


def test_generic_scanner_never_claims_the_registered_h12core_identity(tmp_path):
    adapter = construct(stand_in(tmp_path, QUIET))
    identity = adapter.provenance
    assert identity["method_id"] == fm.GENERIC_METHOD_ID != fm.BASELINE_ID
    assert "baseline_id" not in identity and identity["motif_collection"] == "caller-supplied"
    assert identity["motif_provenance"] == SYNTHETIC
    assert "HOCOMOCO" not in identity["training_overlap"] and SYNTHETIC in identity["training_overlap"]
    with pytest.raises(ValueError, match="motif_provenance"):
        construct(stand_in(tmp_path, QUIET), motif_provenance=" ")


@pytest.mark.skipif(not H12CORE, reason="REWIRE_H12CORE not set")
def test_h12core_identity_differs_from_generic_scan_of_the_same_file(tmp_path):
    executable = stand_in(tmp_path, QUIET)
    registered = fm.H12CoreFimoHitCount(executable, H12CORE, fimo_sha256=sha(executable)).provenance
    generic = construct(executable, motifs=Path(H12CORE), motif_provenance="my copy").provenance
    assert registered["baseline_id"] == registered["method_id"] == fm.BASELINE_ID
    assert "HOCOMOCO" in registered["training_overlap"]
    assert generic["method_id"] == fm.GENERIC_METHOD_ID and "baseline_id" not in generic
    assert registered["configuration_sha256"] != generic["configuration_sha256"]


def test_files_replaced_after_construction_stop_the_scan(tmp_path):
    motifs = tmp_path / "motifs.meme"
    motifs.write_bytes((FIXTURE / "synthetic.meme").read_bytes())
    executable = stand_in(tmp_path, QUIET)
    adapter = construct(executable, motifs=motifs)
    assert adapter.predict(inputs(["none"])) == {"none": 0}
    motifs.write_bytes(motifs.read_bytes().replace(b"SYN_PAL", b"SYN_XXX"))
    with pytest.raises(RuntimeError, match="motifs.meme changed"):
        adapter.predict(inputs(["none"]))
    adapter = construct(executable, motifs=(FIXTURE / "synthetic.meme"))
    executable.write_text(executable.read_text() + "\n# replaced\n")
    with pytest.raises(RuntimeError, match="fimo changed"):
        adapter.predict(inputs(["none"]))


def test_bare_executable_name_is_resolved_not_searched_on_path(tmp_path, monkeypatch):
    local = tmp_path / "work"
    local.mkdir()
    on_path = tmp_path / "bin"
    on_path.mkdir()
    stand_in(local, QUIET)                    # ./fimo: the file that is hashed
    stand_in(on_path, "exit 7")               # a different 5.5.9 fimo earlier on PATH
    monkeypatch.chdir(local)
    monkeypatch.setenv("PATH", f"{on_path}{os.pathsep}{os.environ['PATH']}")
    adapter = fm.FimoHitCount("fimo", FIXTURE / "synthetic.meme", fimo_sha256=sha(local / "fimo"),
                              motifs_sha256=sha(FIXTURE / "synthetic.meme"), motif_provenance=SYNTHETIC)
    assert adapter.fimo == (local / "fimo").resolve()
    assert adapter.predict(inputs(["none"])) == {"none": 0}  # the PATH copy would exit 7
    with pytest.raises(ValueError, match="fimo_sha256"):
        fm.FimoHitCount("fimo", FIXTURE / "synthetic.meme", fimo_sha256=sha(on_path / "fimo"),
                        motifs_sha256=sha(FIXTURE / "synthetic.meme"), motif_provenance=SYNTHETIC)
