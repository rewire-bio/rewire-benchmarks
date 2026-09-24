"""Canonical selection and cross-format checks on small synthetic GTF fixtures."""
import csv
import gzip
import os

import pytest
from mfass import matched_annotation as ma

TAG = '; tag "Ensembl_canonical"'


def _row(chrom, feature, start, end, strand, attributes):
    return "\t".join([chrom, "TEST", feature, str(start), str(end), ".", strand, ".",
                      attributes]) + "\n"


def _gene(gene_id, chrom, start, end, strand):
    return _row(chrom, "gene", start, end, strand,
                f'gene_id "{gene_id}"; gene_type "protein_coding"; gene_name "{gene_id}N";')


def _tx(gene_id, tx, chrom, exons, strand, canonical=True, exon_tags=None):
    """Transcript row plus exons; exon_tags overrides tagging per exon."""
    tag = TAG if canonical else ""
    base = f'gene_id "{gene_id}"; transcript_id "{tx}"; gene_type "protein_coding"'
    rows = [_row(chrom, "transcript", exons[0][0], exons[-1][1], strand, base + tag + ";")]
    ordered = exons if strand == "+" else exons[::-1]  # GENCODE lists exons 5' to 3'
    for n, (start, end) in enumerate(ordered):
        exon_tag = tag if exon_tags is None else (TAG if exon_tags[n] else "")
        rows.append(_row(chrom, "exon", start, end, strand,
                         f'{base}; exon_number {n + 1}{exon_tag};'))
        rows.append(_row(chrom, "CDS", start, end, strand, base + tag + ";"))
    return rows


def fixture_rows():
    return [
        "##description: synthetic\n",
        # Positive strand gene; canonical transcript narrower than the gene.
        _gene("G1.1", "chr1", 100, 1000, "+"),
        *_tx("G1.1", "T1a.1", "chr1", [(150, 200), (300, 400), (600, 900)], "+"),
        *_tx("G1.1", "T1b.1", "chr1", [(100, 200), (300, 1000)], "+", canonical=False),
        # Negative strand gene overlapping G1.
        _gene("G2.3", "chr1", 350, 1200, "-"),
        *_tx("G2.3", "T2.3", "chr1", [(350, 500), (1100, 1200)], "-"),
        # Second positive-strand gene overlapping G1, for same-strand overlap.
        _gene("G5.1", "chr1", 380, 700, "+"),
        *_tx("G5.1", "T5.1", "chr1", [(380, 450), (650, 700)], "+"),
        # Gene without a canonical transcript: excluded and recorded.
        _gene("G3.1", "chr1", 2000, 2100, "+"),
        *_tx("G3.1", "T3.1", "chr1", [(2000, 2100)], "+", canonical=False),
        # Single-exon negative-strand gene on another contig.
        _gene("G4.1", "chr2", 10, 60, "-"),
        *_tx("G4.1", "T4.1", "chr2", [(10, 60)], "-"),
    ]


def write_gtf(path, rows, compress=False):
    opener = gzip.open if compress else open
    with opener(path, "wt") as handle:
        handle.writelines(rows)
    return path


def write_reference(tmp_path, contigs=("chr1", "chr2"), length=1300, width=60, lower=()):
    fasta, fai = tmp_path / "ref.fa", tmp_path / "ref.fa.fai"
    offset, index = 0, []
    with fasta.open("w") as handle:
        for n, name in enumerate(contigs):
            header = f">{name}\n"
            handle.write(header)
            offset += len(header)
            sequence = "ACGT"[n] * length
            if name in lower:
                sequence = sequence.lower()
            lines = [sequence[i:i + width] for i in range(0, length, width)]
            handle.write("\n".join(lines) + "\n")
            index.append(f"{name}\t{length}\t{offset}\t{width}\t{width + 1}\n")
            offset += len("\n".join(lines)) + 1
    fai.write_text("".join(index))
    return fasta, fai


@pytest.mark.parametrize("compress", [False, True])
def test_selection_uses_only_the_canonical_transcript(tmp_path, compress):
    gtf = write_gtf(tmp_path / ("a.gtf.gz" if compress else "a.gtf"), fixture_rows(), compress)
    selected, exclusions = ma.select_canonical(gtf)
    assert [g.gene_id for g in selected] == ["G1.1", "G2.3", "G5.1", "G4.1"]
    g1 = selected[0]
    assert (g1.transcript_id, g1.tx_start, g1.tx_end) == ("T1a.1", 150, 900)
    assert (g1.gtf_gene_start, g1.gtf_gene_end) == (100, 1000)
    assert g1.exons == [(150, 200), (300, 400), (600, 900)]
    g2 = selected[1]
    assert (g2.strand, g2.exons) == ("-", [(350, 500), (1100, 1200)])
    assert exclusions == [{"gene_id": "G3.1", "chrom": "chr1", "strand": "+",
                           "gtf_gene_start": 2000, "gtf_gene_end": 2100,
                           "reason": "no Ensembl_canonical transcript"}]


def test_multiple_canonical_transcripts_stop_preparation(tmp_path):
    rows = fixture_rows() + _tx("G1.1", "T1c.1", "chr1", [(160, 210)], "+")
    with pytest.raises(ma.AmbiguousCanonical, match="G1.1"):
        ma.select_canonical(write_gtf(tmp_path / "a.gtf", rows))


@pytest.mark.parametrize("change, message", [
    (lambda rows: rows + [_row("chr1", "exon", 950, 960, "+",
                               f'gene_id "G1.1"; transcript_id "T1a.1"{TAG};')],
     "span differs from exon envelope"),
    (lambda rows: [r.replace('exon_number 2; tag "Ensembl_canonical"', "exon_number 2")
                   if '"T1a.1"' in r else r for r in rows], "lack the Ensembl_canonical"),
    (lambda rows: rows + [_row("chr1", "exon", 5, 6, "+",
                               f'gene_id "G3.1"; transcript_id "T3.1"{TAG};')],
     "whose transcript was not selected"),
    (lambda rows: rows + [_gene("G1.1", "chr1", 100, 1000, "+")], "duplicate gene row"),
])
def test_inconsistent_annotations_stop_preparation(tmp_path, change, message):
    with pytest.raises(ma.SelectionError, match=message):
        ma.select_canonical(write_gtf(tmp_path / "a.gtf", change(fixture_rows())))


def test_spliceai_table_uses_zero_based_starts_on_both_strands(tmp_path):
    selected, _ = ma.select_canonical(write_gtf(tmp_path / "a.gtf", fixture_rows()))
    ma.write_spliceai_table(selected, tmp_path / "spliceai.tsv")
    with (tmp_path / "spliceai.tsv").open() as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    assert rows[0] == ma.SPLICEAI_HEADER
    assert rows[1] == ["G1.1", "chr1", "+", "149", "900", "149,299,599,", "200,400,900,"]
    assert rows[2] == ["G2.3", "chr1", "-", "349", "1200", "349,1099,", "500,1200,"]
    assert rows[4] == ["G4.1", "chr2", "-", "9", "60", "9,", "60,"]
    parsed = ma.read_spliceai_table(tmp_path / "spliceai.tsv")  # SpliceAI's own parsing rule
    assert list(parsed["tx_starts"]) == [150, 350, 380, 10]
    assert list(parsed["exon_starts"][1]) == [350, 1100]
    assert list(parsed["exon_ends"][1]) == [500, 1200]


def test_selected_gtf_replaces_only_gene_span(tmp_path):
    selected, _ = ma.select_canonical(write_gtf(tmp_path / "a.gtf", fixture_rows()))
    ma.write_selected_gtf(selected, tmp_path / "p.gtf")
    lines = (tmp_path / "p.gtf").read_text().splitlines()
    genes = [line.split("\t") for line in lines if line.split("\t")[2] == "gene"]
    assert [(g[3], g[4]) for g in genes] == [("150", "900"), ("350", "1200"), ("380", "700"),
                                             ("10", "60")]
    assert not any('"T1b.1"' in line or '"G3.1"' in line or "\tCDS\t" in line for line in lines)
    source = set(fixture_rows())
    assert all(line + "\n" in source for line in lines if "\tgene\t" not in line)


@pytest.fixture
def prepared(tmp_path):
    pytest.importorskip("gffutils")
    gtf = write_gtf(tmp_path / "a.gtf", fixture_rows())
    fasta, fai = write_reference(tmp_path)
    out = tmp_path / "annotation"
    summary = ma.prepare(gtf, out, fai, "synthetic")
    return out, fasta, fai, summary


def test_prepare_writes_both_formats_and_they_agree(prepared):
    out, _, fai, summary = prepared
    report = ma.cross_check(out, fai)
    assert report["passed"], report["mismatches"]
    assert report["genes_checked"] == 4 and report["exons_checked"] == 8
    assert report["strands"] == {"+": 2, "-": 2}
    assert report["contigs"] == {"chr1": 3, "chr2": 1}
    assert summary["genes_excluded"] == 1 and summary["genes_with_span_changed"] == 1
    assert summary["labels_read"] is False and summary["cross_check_passed"] is True
    with pytest.raises(FileExistsError):
        ma.prepare(out / "../a.gtf", out, fai)


@pytest.mark.parametrize("target, old, new, check", [
    ("spliceai.tsv", "G1.1\tchr1\t+\t149", "G1.1\tchr1\t+\t150", "spliceai_span_start"),
    ("spliceai.tsv", "149,299,599,", "149,300,599,", "spliceai_exons"),
    ("spliceai.tsv", "G2.3\tchr1\t-", "G2.3\tchr1\t+", "spliceai_strand"),
    ("selection.tsv", "\tchr2\t-\t", "\tchr2\t+\t", "pangolin_strand"),
    ("selection.tsv", "\t350,1100\t", "\t350,1101\t", "pangolin_exon_boundaries"),
    ("selection.tsv", "\t380\t700\t2\t", "\t380\t701\t2\t", "pangolin_span_end"),
])
def test_cross_check_detects_any_disagreement(prepared, target, old, new, check):
    out, _, fai, _ = prepared
    path = out / target
    text = path.read_text()
    assert text.count(old) == 1
    path.write_text(text.replace(old, new))
    report = ma.cross_check(out, fai)
    assert not report["passed"]
    assert check in {m["check"] for m in report["mismatches"]}


def test_missing_reference_contig_is_rejected(tmp_path):
    pytest.importorskip("gffutils")
    gtf = write_gtf(tmp_path / "a.gtf", fixture_rows())
    _, fai = write_reference(tmp_path, contigs=("chr1",))
    with pytest.raises(ma.SelectionError, match="chr2"):
        ma.prepare(gtf, tmp_path / "annotation", fai)
    assert not (tmp_path / "annotation").exists()


def test_fai_reader_crosses_line_boundaries(tmp_path):
    fasta, fai = write_reference(tmp_path, width=7)
    reader = ma.FaiReader(fasta, fai)
    assert reader.fetch("chr2", 5, 20) == "C" * 16
    assert reader.fetch("chr1", 1300, 1300) == "A"
    with pytest.raises(IndexError):
        reader.fetch("chr1", 1300, 1301)


CHECKOUT = os.environ.get("MFASS_PANGOLIN_CHECKOUT")


@pytest.mark.skipif(not CHECKOUT, reason="MFASS_PANGOLIN_CHECKOUT not set")
def test_variant_check_uses_each_tools_gene_rule_without_labels(prepared, tmp_path):
    import subprocess

    out, fasta, _, _ = prepared
    source = tmp_path / "pangolin.py"
    source.write_text(subprocess.run(
        ["git", "-C", CHECKOUT, "show",
         "5cf94b8db938c658391b4305cd7ce33297d44ff7:pangolin/pangolin.py"],
        check=True, capture_output=True, text=True).stdout)
    cases = [  # id, chrom, position, reference allele
        ("same_strand_overlap", "chr1", 390, "A"),  # G1.1, G5.1 (+) and G2.3 (-)
        ("outside_transcript", "chr1", 120, "A"),   # inside G1 gene row, outside T1a span
        ("tx_first_base_plus", "chr1", 150, "A"),
        ("tx_last_base_plus", "chr1", 900, "A"),
        ("tx_first_base_minus", "chr2", 10, "C"),
        ("tx_last_base_minus", "chr2", 60, "C"),
        ("reference_mismatch", "chr1", 395, "G"),
        ("no_coordinate", "chr1", "NA", "A"),
    ]
    cohort, split = tmp_path / "cohort.tsv", tmp_path / "split.tsv"
    with cohort.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["id", "chr", "snp_position_hg38_1based", "ref_allele", "alt_allele",
                         "sdv"])
        writer.writerows([[i, c, p, r, "T", "LABEL-MUST-NOT-BE-READ"] for i, c, p, r in cases]
                         + [["train_only", "chr1", 390, "A", "T", "1"]])
    with split.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["id", "group", "split"])
        writer.writerows([[i, i, "test"] for i, *_ in cases] + [["train_only", "g", "train"]])
    rows, fields, summary = ma.variant_check(out, cohort, split, fasta, source)
    assert "sdv" not in fields and summary["labels_read"] is False
    got = {r["id"]: r for r in rows}
    assert set(got) == {c[0] for c in cases}
    overlap = got["same_strand_overlap"]
    assert overlap["spliceai_genes"] == overlap["pangolin_genes"] == "G1.1,G2.3,G5.1"
    assert overlap["reference_matches_spliceai_rule"] == "True"
    assert got["outside_transcript"]["spliceai_genes"] == ""
    assert got["outside_transcript"]["pangolin_genes"] == ""
    assert got["reference_mismatch"]["reference_matches_pangolin_rule"] == "False"
    assert got["no_coordinate"]["status"] == "no hg38 coordinate"
    assert summary["multiple_genes_same_strand_pangolin"] == 2
    # SpliceAI includes both span ends. Pangolin's get_genes queries
    # region(pos - 1, pos - 1), so it omits a variant on a gene's lowest
    # coordinate on either strand. This is upstream behaviour, reported by the
    # check rather than corrected.
    boundary = {c: (got[c]["spliceai_genes"], got[c]["pangolin_genes"]) for c in
                ("tx_first_base_plus", "tx_last_base_plus", "tx_first_base_minus",
                 "tx_last_base_minus")}
    assert boundary == {"tx_first_base_plus": ("G1.1", ""),
                        "tx_last_base_plus": ("G1.1,G2.3", "G1.1,G2.3"),
                        "tx_first_base_minus": ("G4.1", ""),
                        "tx_last_base_minus": ("G4.1", "G4.1")}
    assert summary["gene_sets_differ"] == 2


@pytest.mark.skipif(not CHECKOUT, reason="MFASS_PANGOLIN_CHECKOUT not set")
def test_reference_case_rules_are_reported_separately(tmp_path):
    """SpliceAI upper-cases the reference; Pangolin compares the bases as read."""
    import subprocess

    pytest.importorskip("gffutils")
    gtf = write_gtf(tmp_path / "a.gtf", fixture_rows())
    fasta, fai = write_reference(tmp_path, lower=("chr2",))
    out = tmp_path / "annotation"
    ma.prepare(gtf, out, fai, "synthetic", fasta)
    source = tmp_path / "pangolin.py"
    source.write_text(subprocess.run(
        ["git", "-C", CHECKOUT, "show",
         "5cf94b8db938c658391b4305cd7ce33297d44ff7:pangolin/pangolin.py"],
        check=True, capture_output=True, text=True).stdout)
    cohort, split = tmp_path / "cohort.tsv", tmp_path / "split.tsv"
    cohort.write_text("id\tchr\tsnp_position_hg38_1based\tref_allele\talt_allele\n"
                      "soft\tchr2\t30\tC\tT\nhard\tchr1\t390\tA\tT\n")
    split.write_text("id\tgroup\tsplit\nsoft\tg1\ttest\nhard\tg2\ttest\n")
    rows, _, summary = ma.variant_check(out, cohort, split, fasta, source)
    got = {r["id"]: r for r in rows}
    assert got["soft"]["reference_matches_spliceai_rule"] == "True"
    assert got["soft"]["reference_matches_pangolin_rule"] == "False"
    assert got["hard"]["reference_matches_pangolin_rule"] == "True"
    assert summary["reference_mismatch_spliceai_rule"] == 0
    assert summary["reference_mismatch_pangolin_rule"] == 1
    assert summary["scoreable_spliceai"] == 2 and summary["scoreable_pangolin"] == 1
    summary_json = __import__("json").loads((out / "summary.json").read_text())
    assert summary_json["reference_fasta"]["sha256"] == ma.file_sha256(fasta)
