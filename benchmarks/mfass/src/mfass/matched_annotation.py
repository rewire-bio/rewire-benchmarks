"""One canonical-transcript selection, written as SpliceAI and Pangolin annotations.

Protocol (docs/mfass-specialist-comparison.md, preparation gates 3-5):

- Select transcripts tagged `Ensembl_canonical` once from the GENCODE v44
  primary-assembly GTF. A gene with more than one stops preparation; a gene
  with none is excluded and recorded. Selection never reads MFASS labels.
- Write both formats from that one selection, keyed by stable versioned gene ID.
- Gene span rule: the Pangolin database gene span is set to the selected
  transcript span, matching SpliceAI's TX_START/TX_END eligibility. This
  departs from upstream create_db.py, which keeps the GTF gene span.
- Coordinates: GTF is 1-based closed. The SpliceAI table stores 0-based starts
  and unchanged ends; its reader adds one to starts.
- Cross-check every gene across the selection manifest, the SpliceAI table (read
  the way SpliceAI reads it) and the gffutils database (read the way Pangolin's
  get_genes reads it). Any mismatch fails the gate.
"""
import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path

from mfass.specialist_provenance import file_sha256

CANONICAL_TAG = "Ensembl_canonical"
SPAN_RULE = ("Gene span is the selected Ensembl_canonical transcript span (GTF 1-based, "
             "closed) in both formats")
SPLICEAI_HEADER = ["#NAME", "CHROM", "STRAND", "TX_START", "TX_END", "EXON_START", "EXON_END"]
MANIFEST_FIELDS = ["gene_id", "gene_name", "gene_type", "transcript_id", "chrom", "strand",
                   "gtf_gene_start", "gtf_gene_end", "tx_start", "tx_end", "exon_count",
                   "exon_starts", "exon_ends"]
# gffutils parameters of upstream scripts/create_db.py at Pangolin 5cf94b8.
PANGOLIN_DB_OPTIONS = {"disable_infer_genes": True, "disable_infer_transcripts": True}

_ATTRIBUTE = re.compile(r'\s*([^\s"]+)\s+"([^"]*)"\s*;?')


class SelectionError(ValueError):
    """The GTF cannot yield an unambiguous, internally consistent selection."""


class AmbiguousCanonical(SelectionError):
    """At least one gene has more than one Ensembl_canonical transcript."""


@dataclass
class Selected:
    gene_id: str
    gene_name: str
    gene_type: str
    transcript_id: str
    chrom: str
    strand: str
    gtf_gene_start: int
    gtf_gene_end: int
    tx_start: int
    tx_end: int
    exons: list = field(default_factory=list)          # [(start, end)], 1-based closed
    gene_line: str = ""
    transcript_line: str = ""
    exon_lines: list = field(default_factory=list)


def parse_attributes(text):
    """GTF attributes as name -> list of values (tags repeat)."""
    values = defaultdict(list)
    for name, value in _ATTRIBUTE.findall(text):
        values[name].append(value)
    return values


def _open_text(path):
    import gzip
    path = Path(path)
    return gzip.open(path, "rt") if path.suffix == ".gz" else path.open()


def select_canonical(gtf_path, tag=CANONICAL_TAG):
    """Return (selected genes in GTF order, exclusions). Raise on any inconsistency."""
    genes, canonical = {}, defaultdict(list)
    tagged_exons, untagged_exon_transcripts = defaultdict(list), set()
    problems = []
    with _open_text(gtf_path) as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            columns = line.rstrip("\n").split("\t")
            if len(columns) != 9:
                raise SelectionError(f"Malformed GTF line: {line[:80]!r}")
            chrom, _, feature, start, end, _, strand, _, attributes = columns
            if feature not in ("gene", "transcript", "exon"):
                continue
            attrs = parse_attributes(attributes)
            gene_id = attrs["gene_id"][0]
            tagged = tag in attrs.get("tag", [])
            if feature == "gene":
                if gene_id in genes:
                    problems.append(f"duplicate gene row {gene_id}")
                genes[gene_id] = (chrom, strand, int(start), int(end),
                                  attrs.get("gene_name", [""])[0],
                                  attrs.get("gene_type", [""])[0], line)
            elif feature == "transcript" and tagged:
                canonical[gene_id].append((attrs["transcript_id"][0], chrom, strand,
                                           int(start), int(end), line))
            elif feature == "exon":
                transcript_id = attrs["transcript_id"][0]
                if tagged:
                    tagged_exons[transcript_id].append((chrom, strand, int(start), int(end),
                                                        line))
                else:
                    untagged_exon_transcripts.add(transcript_id)

    ambiguous = {g: [t[0] for t in ts] for g, ts in canonical.items() if len(ts) > 1}
    if ambiguous:
        raise AmbiguousCanonical(
            f"{len(ambiguous)} genes have multiple {tag} transcripts; stop for review: "
            + json.dumps(dict(sorted(ambiguous.items())[:20])))
    orphans = sorted(set(canonical) - set(genes))
    if orphans:
        problems.append(f"{tag} transcripts without a gene row: {orphans[:20]}")

    selected, exclusions, used = [], [], set()
    for gene_id, (chrom, strand, g_start, g_end, name, gtype, gene_line) in genes.items():
        if gene_id not in canonical:
            exclusions.append({"gene_id": gene_id, "chrom": chrom, "strand": strand,
                               "gtf_gene_start": g_start, "gtf_gene_end": g_end,
                               "reason": f"no {tag} transcript"})
            continue
        transcript_id, t_chrom, t_strand, t_start, t_end, t_line = canonical[gene_id][0]
        used.add(transcript_id)
        exons = sorted(tagged_exons.get(transcript_id, []), key=lambda e: (e[2], e[3]))
        where = f"{gene_id}/{transcript_id}"
        if (t_chrom, t_strand) != (chrom, strand):
            problems.append(f"{where}: transcript contig/strand differs from gene")
        if not exons:
            problems.append(f"{where}: no {tag} exons")
            continue
        if transcript_id in untagged_exon_transcripts:
            problems.append(f"{where}: some exons lack the {tag} tag")
        if any((e[0], e[1]) != (chrom, strand) for e in exons):
            problems.append(f"{where}: exon contig/strand differs from gene")
        if any(b[2] <= a[3] for a, b in pairwise(exons)):
            problems.append(f"{where}: duplicate or overlapping exons")
        if (exons[0][2], max(e[3] for e in exons)) != (t_start, t_end):
            problems.append(f"{where}: transcript span differs from exon envelope")
        if not g_start <= t_start <= t_end <= g_end:
            problems.append(f"{where}: transcript extends beyond gene")
        selected.append(Selected(
            gene_id=gene_id, gene_name=name, gene_type=gtype, transcript_id=transcript_id,
            chrom=chrom, strand=strand, gtf_gene_start=g_start, gtf_gene_end=g_end,
            tx_start=t_start, tx_end=t_end, exons=[(e[2], e[3]) for e in exons],
            gene_line=gene_line, transcript_line=t_line, exon_lines=[e[4] for e in exons]))
    stray = sorted(set(tagged_exons) - used)
    if stray:
        problems.append(f"{tag} exons whose transcript was not selected: {stray[:20]}")
    if problems:
        raise SelectionError(f"{len(problems)} selection problems: " + "; ".join(problems[:20]))
    return selected, exclusions


def read_fai_contigs(fai_path):
    with Path(fai_path).open() as handle:
        return {line.split("\t", 1)[0] for line in handle if line.strip()}


def _manifest_row(gene):
    return {**{key: getattr(gene, key) for key in MANIFEST_FIELDS
               if key not in ("exon_count", "exon_starts", "exon_ends")},
            "exon_count": len(gene.exons),
            "exon_starts": ",".join(str(s) for s, _ in gene.exons),
            "exon_ends": ",".join(str(e) for _, e in gene.exons)}


def write_spliceai_table(selected, path):
    """SpliceAI 1.3.1 custom annotation: 0-based starts, retained ends, trailing commas."""
    with Path(path).open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(SPLICEAI_HEADER)
        for g in selected:
            writer.writerow([g.gene_id, g.chrom, g.strand, g.tx_start - 1, g.tx_end,
                             "".join(f"{s - 1}," for s, _ in g.exons),
                             "".join(f"{e}," for _, e in g.exons)])


def write_selected_gtf(selected, path):
    """Selected rows verbatim, except gene start/end replaced by the transcript span."""
    with Path(path).open("w") as handle:
        for g in selected:
            columns = g.gene_line.rstrip("\n").split("\t")
            columns[3], columns[4] = str(g.tx_start), str(g.tx_end)
            handle.write("\t".join(columns) + "\n")
            handle.write(g.transcript_line)
            handle.writelines(g.exon_lines)


def build_pangolin_db(gtf_path, db_path):
    import gffutils

    gffutils.create_db(str(gtf_path), str(db_path), **PANGOLIN_DB_OPTIONS)


def read_spliceai_table(path):
    """Parse exactly as spliceai.utils.Annotator 1.3.1 does, returning 1-based starts."""
    import numpy as np
    import pandas as pd

    df = pd.read_csv(path, sep="\t", dtype={"CHROM": object})
    return {
        "genes": df["#NAME"].to_numpy(), "chroms": df["CHROM"].to_numpy(),
        "strands": df["STRAND"].to_numpy(),
        "tx_starts": df["TX_START"].to_numpy() + 1, "tx_ends": df["TX_END"].to_numpy(),
        "exon_starts": [np.asarray(list(map(int, re.split(",", c)[:-1]))) + 1
                        for c in df["EXON_START"].to_numpy()],
        "exon_ends": [np.asarray(list(map(int, re.split(",", c)[:-1])))
                      for c in df["EXON_END"].to_numpy()],
    }


def read_manifest(path):
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    for row in rows:
        for key in ("gtf_gene_start", "gtf_gene_end", "tx_start", "tx_end", "exon_count"):
            row[key] = int(row[key])
        row["exons"] = list(zip(map(int, row["exon_starts"].split(",")),
                                map(int, row["exon_ends"].split(","))))
    return rows


def cross_check(directory, fai_path=None):
    """Compare every gene across manifest, SpliceAI table and Pangolin database."""
    import gffutils

    directory = Path(directory)
    manifest = read_manifest(directory / "selection.tsv")
    expected = {r["gene_id"]: r for r in manifest}
    mismatches = []

    def record(kind, gene_id, detail):
        mismatches.append({"check": kind, "gene_id": gene_id, "detail": detail})

    if len(expected) != len(manifest):
        record("manifest_unique_gene_ids", "*", "duplicate gene IDs in selection manifest")

    table = read_spliceai_table(directory / "spliceai.tsv")
    spliceai_ids = list(table["genes"])
    if len(spliceai_ids) != len(set(spliceai_ids)):
        record("spliceai_unique_gene_ids", "*", "duplicate #NAME rows")
    if set(spliceai_ids) != set(expected):
        record("spliceai_gene_ids", "*", {"missing": sorted(set(expected) - set(spliceai_ids))[:20],
                                          "extra": sorted(set(spliceai_ids) - set(expected))[:20]})
    for i, gene_id in enumerate(spliceai_ids):
        row = expected.get(gene_id)
        if row is None:
            continue
        observed = (table["chroms"][i], table["strands"][i],
                    int(table["tx_starts"][i]), int(table["tx_ends"][i]),
                    sorted(zip(map(int, table["exon_starts"][i]), map(int, table["exon_ends"][i]))))
        wanted = (row["chrom"], row["strand"], row["tx_start"], row["tx_end"], row["exons"])
        for name, a, b in zip(("chrom", "strand", "span_start", "span_end", "exons"),
                              observed, wanted):
            if a != b:
                record(f"spliceai_{name}", gene_id, {"spliceai": str(a)[:200],
                                                     "manifest": str(b)[:200]})

    db = gffutils.FeatureDB(str(directory / "pangolin.db"))
    counts = {kind: db.count_features_of_type(kind) for kind in ("gene", "transcript", "exon")}
    wanted_counts = {"gene": len(manifest), "transcript": len(manifest),
                     "exon": sum(r["exon_count"] for r in manifest)}
    if counts != wanted_counts:
        record("pangolin_feature_counts", "*", {"db": counts, "manifest": wanted_counts})
    seen = set()
    for gene in db.features_of_type("gene"):
        gene_id = gene["gene_id"][0]  # the key Pangolin's get_genes reports
        seen.add(gene_id)
        row = expected.get(gene_id)
        if row is None:
            record("pangolin_gene_ids", gene_id, "gene not in selection")
            continue
        transcripts = [t["transcript_id"][0] for t in db.children(gene, featuretype="transcript")]
        # Pangolin's get_genes collects gene.children(featuretype="exon") boundaries.
        boundaries = []
        for exon in db.children(gene, featuretype="exon"):
            boundaries.extend([exon[3], exon[4]])
        wanted_boundaries = [p for pair in row["exons"] for p in pair]
        checks = (("chrom", gene.seqid, row["chrom"]), ("strand", gene.strand, row["strand"]),
                  ("span_start", gene.start, row["tx_start"]),
                  ("span_end", gene.end, row["tx_end"]),
                  ("transcripts", transcripts, [row["transcript_id"]]),
                  ("exon_boundaries", sorted(boundaries), sorted(wanted_boundaries)))
        for name, a, b in checks:
            if a != b:
                record(f"pangolin_{name}", gene_id, {"pangolin": str(a)[:200],
                                                     "manifest": str(b)[:200]})
    if seen != set(expected):
        record("pangolin_gene_ids", "*", {"missing": sorted(set(expected) - seen)[:20]})

    contigs = Counter(r["chrom"] for r in manifest)
    missing_contigs = []
    if fai_path is not None:
        missing_contigs = sorted(set(contigs) - read_fai_contigs(fai_path))
        for contig in missing_contigs:
            record("reference_contig", "*", f"{contig} absent from reference index")
    strands = Counter(r["strand"] for r in manifest)
    return {
        "passed": not mismatches,
        "genes_checked": len(manifest),
        "exons_checked": wanted_counts["exon"],
        "pangolin_feature_counts": counts,
        "strands": dict(strands),
        "contigs": dict(sorted(contigs.items())),
        "reference_index_checked": fai_path is not None,
        "missing_contigs": missing_contigs,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:200],
    }


class FaiReader:
    """Uncompressed FASTA access through its .fai index, without extra dependencies."""

    def __init__(self, fasta_path, fai_path=None):
        self.path = Path(fasta_path)
        self.index = {}
        with Path(fai_path or f"{fasta_path}.fai").open() as handle:
            for line in handle:
                name, length, offset, bases, width = line.split("\t")[:5]
                self.index[name] = tuple(map(int, (length, offset, bases, width)))
        self.handle = self.path.open("rb")

    def fetch(self, chrom, start, end):
        """1-based closed interval."""
        length, offset, bases, width = self.index[chrom]
        if not 1 <= start <= end <= length:
            raise IndexError(f"{chrom}:{start}-{end} outside contig")
        first = offset + (start - 1) // bases * width + (start - 1) % bases
        last = offset + (end - 1) // bases * width + (end - 1) % bases
        self.handle.seek(first)
        return self.handle.read(last - first + 1).replace(b"\n", b"").replace(
            b"\r", b"").decode("ascii")


def load_upstream_get_genes(pangolin_source):
    """Pangolin's own get_genes, taken from its source without importing torch."""
    import ast

    tree = ast.parse(Path(pangolin_source).read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == "get_genes")
    namespace = {}
    # Runs one function definition from the pinned upstream file; nothing else is executed.
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(pangolin_source), "exec"),  # noqa: S102
         namespace)
    return namespace["get_genes"]


def variant_check(directory, cohort, split, fasta, pangolin_source, fai_path=None):
    """Per test variant: genes each tool would consider, and the reference allele.

    Uses only identifiers, coordinates and alleles; outcome columns are not read.
    SpliceAI eligibility follows Annotator.get_name_and_strand (same contig,
    TX_START+1 <= pos <= TX_END). Pangolin eligibility runs its get_genes.
    """
    import gffutils

    directory = Path(directory)
    with Path(split).open(newline="") as handle:
        test_ids = {r["id"] for r in csv.DictReader(handle, delimiter="\t")
                    if r["split"] == "test"}
    wanted = ("id", "chr", "snp_position_hg38_1based", "ref_allele", "alt_allele")
    with Path(cohort).open(newline="") as handle:
        rows = [{k: r[k] for k in wanted} for r in csv.DictReader(handle, delimiter="\t")
                if r["id"] in test_ids]
    table = read_spliceai_table(directory / "spliceai.tsv")
    by_chrom = defaultdict(list)
    for i, chrom in enumerate(table["chroms"]):
        by_chrom[chrom].append(i)
    db = gffutils.FeatureDB(str(directory / "pangolin.db"))
    get_genes = load_upstream_get_genes(pangolin_source)
    reference = FaiReader(fasta, fai_path)
    out, counts = [], Counter()
    for r in rows:
        pos = r["snp_position_hg38_1based"]
        record = {**r, "spliceai_genes": "", "pangolin_genes": "", "genes_agree": "",
                  "reference_matches_spliceai_rule": "", "reference_matches_pangolin_rule": "",
                  "status": ""}
        if pos in ("NA", "", None):
            record["status"] = "no hg38 coordinate"
        elif r["chr"] not in reference.index:
            record["status"] = "contig absent from reference"
        else:
            pos = int(pos)
            spliceai = sorted(str(table["genes"][i]) for i in by_chrom.get(r["chr"], [])
                              if table["tx_starts"][i] <= pos <= table["tx_ends"][i])
            genes_pos, genes_neg = get_genes(r["chr"], pos, db)
            pangolin = sorted([*genes_pos, *genes_neg])
            try:
                observed = reference.fetch(r["chr"], pos, pos + len(r["ref_allele"]) - 1)
            except IndexError:
                observed = None
            # SpliceAI upper-cases the reference before comparing; Pangolin compares as read.
            spliceai_match = observed is not None and observed.upper() == r["ref_allele"]
            pangolin_match = observed is not None and observed == r["ref_allele"]
            record.update(spliceai_genes=",".join(spliceai), pangolin_genes=",".join(pangolin),
                          genes_agree=str(spliceai == pangolin),
                          reference_matches_spliceai_rule=str(spliceai_match),
                          reference_matches_pangolin_rule=str(pangolin_match), status="checked")
            counts["spliceai_has_gene"] += bool(spliceai)
            counts["pangolin_has_gene"] += bool(pangolin)
            counts["gene_sets_differ"] += spliceai != pangolin
            counts["reference_mismatch_spliceai_rule"] += not spliceai_match
            counts["reference_mismatch_pangolin_rule"] += not pangolin_match
            counts["scoreable_spliceai"] += bool(spliceai) and spliceai_match
            counts["scoreable_pangolin"] += bool(pangolin) and pangolin_match
            counts["multiple_genes_same_strand_pangolin"] += (len(genes_pos) > 1 or
                                                               len(genes_neg) > 1)
        counts[record["status"]] += 1
        out.append(record)
    fields = [*wanted, "status", "spliceai_genes", "pangolin_genes", "genes_agree",
              "reference_matches_spliceai_rule", "reference_matches_pangolin_rule"]
    return out, fields, {"test_variants": len(rows), **dict(sorted(counts.items())),
                         "labels_read": False}


def file_md5(path):
    digest = hashlib.md5(usedforsecurity=False)
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tool_versions():
    import platform
    import sqlite3
    from importlib import metadata
    return {"python": platform.python_version(), "sqlite": sqlite3.sqlite_version,
            **{name: metadata.version(name) for name in ("gffutils", "pandas", "numpy")}}


def prepare(gtf_path, out_dir, fai_path=None, annotation_release=None, fasta_path=None):
    """Select, write both formats, cross-check, and record a summary receipt."""
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"Refusing to reuse {out_dir}")
    selected, exclusions = select_canonical(gtf_path)
    if fai_path is not None:
        missing = sorted({g.chrom for g in selected} - read_fai_contigs(fai_path))
        if missing:
            raise SelectionError(f"Selected contigs absent from reference: {missing}")
    out_dir.mkdir(parents=True)
    with (out_dir / "selection.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, delimiter="\t",
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(_manifest_row(g) for g in selected)
    with (out_dir / "exclusions.tsv").open("w", newline="") as handle:
        fields = ["gene_id", "chrom", "strand", "gtf_gene_start", "gtf_gene_end", "reason"]
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(exclusions)
    write_spliceai_table(selected, out_dir / "spliceai.tsv")
    write_selected_gtf(selected, out_dir / "pangolin.gtf")
    build_pangolin_db(out_dir / "pangolin.gtf", out_dir / "pangolin.db")
    report = cross_check(out_dir, fai_path)
    (out_dir / "cross-check.json").write_text(json.dumps(report, indent=2) + "\n")

    changed = sum((g.gtf_gene_start, g.gtf_gene_end) != (g.tx_start, g.tx_end) for g in selected)
    summary = {
        "gtf": {"file": Path(gtf_path).name, "sha256": file_sha256(gtf_path),
                "md5": file_md5(gtf_path),
                "annotation_release": annotation_release or "unreported"},
        "reference_index": ({"file": Path(fai_path).name, "sha256": file_sha256(fai_path)}
                            if fai_path else None),
        "reference_fasta": ({"file": Path(fasta_path).name, "sha256": file_sha256(fasta_path)}
                            if fasta_path else None),
        "selection_tag": CANONICAL_TAG,
        "span_rule": SPAN_RULE,
        "departures_from_upstream_create_db": [
            "gene start/end replaced by the selected transcript span",
            "genes without an Ensembl_canonical transcript are dropped, not kept exon-less",
        ],
        "genes_in_gtf": len(selected) + len(exclusions),
        "genes_selected": len(selected),
        "genes_excluded": len(exclusions),
        "genes_with_span_changed": changed,
        "labels_read": False,
        "cross_check_passed": report["passed"],
        "tool_versions": tool_versions(),
        "pangolin_db_options": PANGOLIN_DB_OPTIONS,
        "outputs": {name: file_sha256(out_dir / name) for name in (
            "selection.tsv", "exclusions.tsv", "spliceai.tsv", "pangolin.gtf", "cross-check.json")},
        "note": ("pangolin.db content is verified by cross-check.json and its bytes are hashed "
                 "per run. Byte identity across gffutils or SQLite versions is not assumed."),
    }
    summary["outputs"]["pangolin.db"] = file_sha256(out_dir / "pangolin.db")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not report["passed"]:
        raise SelectionError(f"Cross-format check failed: {report['mismatch_count']} mismatches; "
                             f"see {out_dir / 'cross-check.json'}")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    build = sub.add_parser("prepare")
    build.add_argument("--gtf", required=True)
    build.add_argument("--fai", required=True, help="reference .fai used by both runners")
    build.add_argument("--out", required=True, help="new directory")
    build.add_argument("--annotation-release", default=None)
    build.add_argument("--fasta", default=None, help="reference FASTA, hashed into the summary")
    check = sub.add_parser("check")
    check.add_argument("--dir", required=True)
    check.add_argument("--fai", required=True)
    variants = sub.add_parser("variants", help="label-free per-variant eligibility check")
    variants.add_argument("--dir", required=True)
    variants.add_argument("--cohort", required=True)
    variants.add_argument("--split", required=True)
    variants.add_argument("--fasta", required=True, help="uncompressed FASTA with .fai")
    variants.add_argument("--pangolin-source", required=True, help="installed pangolin.py")
    variants.add_argument("--out", required=True, help="new TSV; a .json summary is written beside it")
    args = ap.parse_args()
    if args.command == "prepare":
        result = prepare(args.gtf, args.out, args.fai, args.annotation_release, args.fasta)
    elif args.command == "check":
        result = cross_check(args.dir, args.fai)
    else:
        out = Path(args.out)
        if out.exists() or out.with_suffix(".json").exists():
            raise FileExistsError(f"Refusing to overwrite {out}")
        rows, fields, result = variant_check(args.dir, args.cohort, args.split, args.fasta,
                                             args.pangolin_source)
        with out.open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t",
                                    lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        from mfass.pangolin_patch import source_identity
        result.update(annotation_dir_summary_sha256=file_sha256(Path(args.dir) / "summary.json"),
                      **source_identity(args.pangolin_source),
                      cohort_sha256=file_sha256(args.cohort), split_sha256=file_sha256(args.split),
                      fasta_sha256=file_sha256(args.fasta), tool_versions=tool_versions(),
                      output_sha256=file_sha256(out))
        out.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if args.command == "check" and not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
