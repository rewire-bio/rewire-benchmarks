"""Regression tests for the Pangolin per-gene masking patch, without inference.

Only `process_variant` is executed, extracted from the pinned upstream source
and its patched copy. Reference retrieval, gene lookup and model scores are
replaced by synthetic fixtures, so the numbers carry no biological meaning.

Pangolin is GPL-3.0 and is not vendored here. Set MFASS_PANGOLIN_CHECKOUT to a
git clone of https://github.com/tkzeng/Pangolin containing the pinned revision.
"""
import ast
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from mfass import pangolin_patch as patch
from mfass.run_pangolin import _max_abs_score
from mfass.specialist_provenance import file_sha256

CHECKOUT = os.environ.get("MFASS_PANGOLIN_CHECKOUT")
needs_source = pytest.mark.skipif(
    not CHECKOUT, reason="MFASS_PANGOLIN_CHECKOUT not set; pinned GPL source is not vendored")

POS, D = 10000, 2  # score window covers genomic positions 9998..10002


def test_patch_file_is_the_reviewed_artifact():
    text = patch.PATCH_FILE.read_text()
    assert file_sha256(patch.PATCH_FILE) == patch.PATCH_SHA256
    assert patch.UPSTREAM_REVISION in text and patch.UPSTREAM_SOURCE_SHA256 in text
    assert patch.PATCHED_VERSION in text and "GPL-3.0" in text
    # Only the documented edit: copy per gene, rename strand arrays, version label.
    added = [line for line in text.splitlines() if line.startswith("+") and
             not line.startswith("+++")]
    assert added == [
        "+    for (genes, strand_loss, strand_gain) in (",
        "+            # rewire mask-per-gene patch: masking below edits these arrays in",
        "+            # place, so each gene gets its own copy of the strand's scores.",
        "+            loss, gain = strand_loss.copy(), strand_gain.copy()",
        '+    version="1.0.2+rewire.maskpergene1",',
    ]


def test_unknown_source_bytes_are_not_treated_as_reviewed(tmp_path):
    source = tmp_path / "pangolin.py"
    source.write_text("# some other pangolin\n")
    assert patch.source_identity(source)["pangolin_source_identity"] == "unrecognised"


@pytest.fixture(scope="module")
def sources(tmp_path_factory):
    if not CHECKOUT:
        pytest.skip("MFASS_PANGOLIN_CHECKOUT not set")
    tree = tmp_path_factory.mktemp("pangolin") / "patched"
    receipt = patch.prepare_patched_tree(CHECKOUT, tree)
    original = patch._git(CHECKOUT, "show", f"{patch.UPSTREAM_REVISION}:pangolin/pangolin.py")
    return SimpleNamespace(original=original,
                           patched=(tree / "pangolin" / "pangolin.py").read_text(),
                           tree=tree, receipt=receipt)


def _process_variant(source, genes_pos, genes_neg, scores, mask):
    """Run the source's own process_variant against synthetic inputs."""
    tree = ast.parse(source)
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "process_variant")
    calls = []

    class Contig:
        def __getitem__(self, window):
            return SimpleNamespace(seq="A" * (window.stop - window.start))

    def compute_score(ref_seq, alt_seq, strand, d, models):
        calls.append(strand)
        loss, gain = scores[strand]
        return np.array(loss, dtype=float), np.array(gain, dtype=float)

    namespace = {
        "np": np,
        "pyfastx": SimpleNamespace(Fasta=lambda path: {"chr1": Contig()}),
        "get_genes": lambda chromosome, position, gtf: (genes_pos, genes_neg),
        "compute_score": compute_score,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<pangolin>", "exec"), namespace)  # noqa: S102
    args = SimpleNamespace(distance=D, score_cutoff=None, mask=mask, score_exons="False",
                           reference_file="synthetic.fa")
    raw = namespace["process_variant"](0, "chr1", POS, "A", "C", None, None, args)
    return raw, calls


def _per_gene(raw):
    return {chunk.split("|", 1)[0]: chunk for chunk in raw.split(",")}


# Two overlapping genes on one strand. geneA annotates positions 9998 and 10002
# (window indices 0 and 4); geneB annotates only 9999 (index 1). The largest
# gain sits at index 0, a site geneB does not annotate.
GENES = {"geneA": [9998, 10002], "geneB": [9999, 9999]}
SCORES = ([-0.2, -0.1, 0.0, -0.3, 0.0], [0.8, 0.7, 0.0, 0.0, 0.1])


def _layout(strand, order, genes=GENES):
    """(genes_pos, genes_neg) with the named genes on one strand, in this order."""
    selected = {name: genes[name] for name in order}
    return (selected, {}) if strand == "+" else ({}, selected)


def test_patched_tree_receipt_pins_every_identity(sources):
    receipt = sources.receipt
    assert receipt["upstream_source_sha256"] == patch.UPSTREAM_SOURCE_SHA256
    assert receipt["patched_source_sha256"] == patch.PATCHED_SOURCE_SHA256
    assert receipt["patch_sha256"] == patch.PATCH_SHA256
    assert set(receipt["upstream_files"]) == set(patch.UPSTREAM_FILES)
    assert all(receipt["upstream_files"][name] == {"sha256": sha, "git_blob": blob}
               for name, (sha, blob) in patch.UPSTREAM_FILES.items())
    assert patch.source_identity(sources.tree / "pangolin" / "pangolin.py") == {
        "pangolin_source_sha256": patch.PATCHED_SOURCE_SHA256,
        "pangolin_source_identity": patch.PATCH_ID}


def test_prepare_refuses_a_modified_patch(tmp_path):
    if not CHECKOUT:
        pytest.skip("MFASS_PANGOLIN_CHECKOUT not set")
    altered = tmp_path / "altered.patch"
    altered.write_text(patch.PATCH_FILE.read_text().replace("copy()", "view()"))
    with pytest.raises(ValueError, match="reviewed patch"):
        patch.prepare_patched_tree(CHECKOUT, tmp_path / "tree", altered)
    assert not (tmp_path / "tree").exists()


@needs_source
def test_original_failure_from_the_documented_receipt(sources):
    """Reproduces masking-order-check.json: 0.7 versus 0.8 by gene order alone."""
    scores = {"+": ([-0.2, -0.1, 0.0, 0.0, 0.0], [0.8, 0.7, 0.0, 0.0, 0.0])}
    genes = {"geneA": [9998, 10002], "geneB": [9999, 10002]}
    outputs = {}
    for order in (("geneA", "geneB"), ("geneB", "geneA")):
        raw, _ = _process_variant(sources.original, {g: genes[g] for g in order}, {},
                                  scores, "True")
        outputs[order] = raw
    assert outputs[("geneA", "geneB")] == ("geneA|-1:0.7|-2:-0.2|Warnings:,"
                                           "geneB|-2:0.0|-2:0.0|Warnings:")
    assert outputs[("geneB", "geneA")] == ("geneB|-2:0.8|-1:-0.1|Warnings:,"
                                           "geneA|-2:0.0|-2:0.0|Warnings:")
    assert {_max_abs_score(raw) for raw in outputs.values()} == {0.7, 0.8}


@needs_source
@pytest.mark.parametrize("strand", ["+", "-"])
def test_original_leaks_masking_between_genes_on_either_strand(sources, strand):
    forward, _ = _process_variant(sources.original, *_layout(strand, ("geneA", "geneB")),
                                  {strand: SCORES}, "True")
    reverse, _ = _process_variant(sources.original, *_layout(strand, ("geneB", "geneA")),
                                  {strand: SCORES}, "True")
    assert _per_gene(forward) != _per_gene(reverse)


@needs_source
@pytest.mark.parametrize("strand", ["+", "-"])
def test_patched_scores_do_not_depend_on_gene_order(sources, strand):
    forward, _ = _process_variant(sources.patched, *_layout(strand, ("geneA", "geneB")),
                                  {strand: SCORES}, "True")
    reverse, _ = _process_variant(sources.patched, *_layout(strand, ("geneB", "geneA")),
                                  {strand: SCORES}, "True")
    assert _per_gene(forward) == _per_gene(reverse)
    assert _max_abs_score(forward) == _max_abs_score(reverse)
    # Each gene's masked scores equal those it receives when it is the only gene.
    for gene in GENES:
        alone, _ = _process_variant(sources.patched, *_layout(strand, (gene,)),
                                    {strand: SCORES}, "True")
        assert _per_gene(forward)[gene] == alone
    # Aggregate over genes equals the maximum over genes scored separately.
    singles = [_max_abs_score(_process_variant(sources.patched, *_layout(strand, (g,)),
                                               {strand: SCORES}, "True")[0]) for g in GENES]
    assert _max_abs_score(forward) == max(singles)


@needs_source
@pytest.mark.parametrize("strand", ["+", "-"])
def test_gene_without_annotated_sites_does_not_clamp_later_genes(sources, strand):
    genes = {"noexons": [], "geneB": [9999, 9999]}
    raw, _ = _process_variant(sources.patched, *_layout(strand, ("noexons", "geneB"), genes),
                              {strand: SCORES}, "True")
    alone, _ = _process_variant(sources.patched, *_layout(strand, ("geneB",), genes),
                                {strand: SCORES}, "True")
    original, _ = _process_variant(sources.original,
                                   *_layout(strand, ("noexons", "geneB"), genes),
                                   {strand: SCORES}, "True")
    assert "NoAnnotatedSitesToMaskForThisGene" in _per_gene(raw)["noexons"]
    assert _per_gene(raw)["geneB"] == alone
    assert _per_gene(original)["geneB"] != alone  # upstream carries the clamp forward


@needs_source
def test_opposite_strand_genes_are_scored_independently_in_both_versions(sources):
    scores = {"+": SCORES, "-": ([-0.5, 0.0, 0.0, 0.0, -0.4], [0.2, 0.0, 0.9, 0.0, 0.0])}
    for source in (sources.original, sources.patched):
        raw, calls = _process_variant(source, {"geneA": GENES["geneA"]},
                                      {"geneB": GENES["geneB"]}, scores, "True")
        assert calls == ["+", "-"]
        assert set(_per_gene(raw)) == {"geneA", "geneB"}
    patched, _ = _process_variant(sources.patched, {"geneA": GENES["geneA"]},
                                  {"geneB": GENES["geneB"]}, scores, "True")
    original, _ = _process_variant(sources.original, {"geneA": GENES["geneA"]},
                                   {"geneB": GENES["geneB"]}, scores, "True")
    assert patched == original


@needs_source
@pytest.mark.parametrize("strand", ["+", "-"])
@pytest.mark.parametrize("order", [("geneA", "geneB"), ("geneB", "geneA")])
def test_unmasked_output_is_byte_identical_to_upstream(sources, strand, order):
    patched, _ = _process_variant(sources.patched, *_layout(strand, order),
                                  {strand: SCORES}, "False")
    original, _ = _process_variant(sources.original, *_layout(strand, order),
                                   {strand: SCORES}, "False")
    assert patched == original


@needs_source
def test_patched_source_differs_from_upstream_only_in_process_variant(sources):
    def functions(source):
        return {node.name: ast.dump(node) for node in ast.parse(source).body
                if isinstance(node, ast.FunctionDef)}

    original, patched = functions(sources.original), functions(sources.patched)
    assert original.keys() == patched.keys()
    assert [name for name in original if original[name] != patched[name]] == ["process_variant"]
    assert Path(sources.tree / "setup.py").read_text().count(patch.PATCHED_VERSION) == 1


def test_pinned_files_cover_the_loaded_ensemble_and_model_code():
    assert set(patch.ENSEMBLE_WEIGHTS) | {"model.py", "__init__.py"} == set(patch.UPSTREAM_FILES)


@needs_source
def test_pinned_blob_ids_and_hashes_equal_the_upstream_commit():
    import hashlib
    import subprocess

    for name, (sha256, blob) in patch.UPSTREAM_FILES.items():
        path = f"{patch.UPSTREAM_REVISION}:pangolin/{name}"
        assert patch._git(CHECKOUT, "rev-parse", path).strip() == blob
        data = subprocess.run(["git", "-C", CHECKOUT, "cat-file", "blob", path],
                              check=True, capture_output=True).stdout
        assert hashlib.sha256(data).hexdigest() == sha256
    assert patch._git(CHECKOUT, "rev-parse",
                      f"{patch.UPSTREAM_REVISION}:pangolin/pangolin.py").strip() == (
        patch.UPSTREAM_SOURCE_BLOB)


def test_require_reviewed_names_every_failure():
    good = {"pangolin_source_identity": patch.PATCH_ID, "pangolin_upstream_files_verified": True,
            "pangolin_upstream_file_mismatches": []}
    patch.require_reviewed(good)
    with pytest.raises(ValueError, match="not pangolin-5cf94b8"):
        patch.require_reviewed({**good, "pangolin_source_identity": "unrecognised"})
    patch.require_reviewed({**good, "pangolin_source_identity": "unrecognised"},
                           allow_unpatched=True)
    with pytest.raises(ValueError, match="models/final.1.0.3.v2"):
        patch.require_reviewed({**good, "pangolin_upstream_files_verified": False,
                                "pangolin_upstream_file_mismatches": ["models/final.1.0.3.v2"]})
    with pytest.raises(ValueError):
        patch.require_reviewed({"pangolin_source_identity": patch.PATCH_ID})


def test_installed_identity_detects_altered_weights(tmp_path, monkeypatch):
    """A synthetic package directory with one weight changed is not verified."""
    import importlib.machinery

    package = tmp_path / "pangolin"
    (package / "models").mkdir(parents=True)
    (package / "pangolin.py").write_text("# stand-in\n")
    for name in patch.UPSTREAM_FILES:
        (package / name).write_bytes(b"not the upstream bytes")
    spec = importlib.machinery.ModuleSpec("pangolin", None, is_package=True)
    spec.submodule_search_locations = [str(package)]
    monkeypatch.setattr(patch.importlib.util, "find_spec", lambda name: spec)
    identity = patch.installed_identity()
    assert identity["pangolin_upstream_files_verified"] is False
    assert set(identity["pangolin_upstream_file_mismatches"]) == set(patch.UPSTREAM_FILES)
    assert identity["pangolin_source_identity"] == "unrecognised"


def test_upstream_licence_text_and_exception_note_ship_with_the_patch():
    directory = patch.PATCH_FILE.parent
    licence = directory / "LICENSE-pangolin-GPL-3.0"
    assert file_sha256(licence) == (
        "3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986")
    note = (directory / "README.md").read_text()
    assert patch.PATCH_FILE.name in note and licence.name in note
    assert "GPL-3.0" in note and "MIT" in note and patch.UPSTREAM_REVISION in note
