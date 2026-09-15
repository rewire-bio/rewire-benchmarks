"""Build the MFASS evaluation cohort from the published per-variant table.

Source: Chong et al., Molecular Cell 2018. Analysis repository KosuriLab/MFASS,
file processed_data/snv/snv_data_clean.txt.

The eligibility filter is asserted against the counts reported in the paper so the
cohort cannot drift silently. Run with --check to verify without writing.
"""
import argparse
import csv
import hashlib
import json
import pathlib
import sys

# Published totals, Chong et al. 2018. See RECONCILIATION in the README.
PUBLISHED_VARIANTS = 27733
PUBLISHED_SDVS = 1050
PUBLISHED_EXONS_MUTANT = 2198

KEEP = [
    "id", "ensembl_id", "chr", "strand",
    "intron1_len", "exon_len", "intron2_len",
    "start_hg38_0based", "end_hg38_0based", "snp_position_hg38_1based",
    "ref_allele", "alt_allele", "rel_position", "rel_position_scaled",
    "label", "category", "strong_lof", "delta_dpsi", "sequence",
]

# Joined from processed_data/snv/snv_func_annot.txt on `id`. The gene is needed for
# grouping (variants in two exons of one gene are not independent) and the
# conservation scores belong in the trivial baseline.
ANNOT = ["ensembl_gene_id", "symbol", "phylop_score", "mean_phastCons_score", "cadd_score"]


def eligible(row):
    """The evaluation cohort: assayed mutants with a usable splice-disruption call.

    category == 'mutant' drops the 2,339 natural and 1,358 control sequences, which
    are not variants under test. strong_lof != 'NA' drops 1,239 mutants whose
    disruption call could not be made from the assay.
    """
    return row["category"] == "mutant" and row["strong_lof"] != "NA"


def load(raw_path):
    with open(raw_path, newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def build(raw_path, annot_path=None):
    rows = load(raw_path)
    annot = {}
    if annot_path and pathlib.Path(annot_path).exists():
        for r in load(annot_path):
            annot[r["id"]] = {k: r.get(k, "NA") for k in ANNOT}
    cohort = [r for r in rows if eligible(r)]
    sdv = sum(1 for r in cohort if r["strong_lof"] == "TRUE")
    exons_mutant = len({r["ensembl_id"] for r in rows if r["category"] == "mutant"})

    checks = {
        "variants": (len(cohort), PUBLISHED_VARIANTS),
        "sdvs": (sdv, PUBLISHED_SDVS),
        "exons_in_mutant_set": (exons_mutant, PUBLISHED_EXONS_MUTANT),
    }
    failures = [f"{k}: got {got}, published {want}" for k, (got, want) in checks.items() if got != want]
    if failures:
        raise SystemExit("Cohort does not reconcile with the published totals:\n  " + "\n  ".join(failures))

    out = []
    for r in cohort:
        rec = {k: r[k] for k in KEEP}
        rec["sdv"] = 1 if r["strong_lof"] == "TRUE" else 0
        rec.update(annot.get(r["id"], {k: "NA" for k in ANNOT}))
        out.append(rec)
    return out, checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="benchmarks/mfass/data/snv_data_clean.txt")
    ap.add_argument("--out", default="benchmarks/mfass/data/cohort.tsv")
    ap.add_argument("--annot", default="benchmarks/mfass/data/snv_func_annot.txt",
                    help="functional annotation table joined on id; optional")
    ap.add_argument("--check", action="store_true", help="verify only, write nothing")
    args = ap.parse_args()

    raw = pathlib.Path(args.raw)
    digest = hashlib.sha256(raw.read_bytes()).hexdigest()
    cohort, checks = build(raw, args.annot)

    prevalence = sum(r["sdv"] for r in cohort) / len(cohort)
    exons = len({r["ensembl_id"] for r in cohort})
    summary = {
        "source_file": str(raw),
        "source_sha256": digest,
        "variants": len(cohort),
        "sdvs": sum(r["sdv"] for r in cohort),
        "prevalence": round(prevalence, 5),
        "exons_in_cohort": exons,
        "genes_in_cohort": len({r["ensembl_gene_id"] for r in cohort
                                if r["ensembl_gene_id"] not in ("NA", "")}),
        "annotated": sum(1 for r in cohort if r["phylop_score"] not in ("NA", "")),
        "reconciliation": {k: {"observed": g, "published": w} for k, (g, w) in checks.items()},
    }
    print(json.dumps(summary, indent=2))

    if args.check:
        return

    out = pathlib.Path(args.out)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cohort[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(cohort)
    with open(out.with_suffix(".summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nwrote {out} and {out.with_suffix('.summary.json')}", file=sys.stderr)


if __name__ == "__main__":
    main()
