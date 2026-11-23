"""Extract temporally held-out ClinVar >=2-star missense SNV cohorts and gnomAD missense reference set.

Outputs under workdir (default: this script's directory):
  raw/clinvar_*.vcf.gz               Archived NCBI ClinVar GRCh38 VCFs
  raw/download_manifest.json         Exact URLs, byte sizes, SHA-256 digests
  cohorts/variants_post_2023_12.tsv  Primary held-out >=2-star missense SNVs (not P/LP or B/LB in 2023-12-26)
  cohorts/variants_post_2025_03.tsv  Sensitivity held-out >=2-star missense SNVs (not P/LP or B/LB in 2025-03-30)
  cohorts/gnomad_ref_missense.tsv    Population missense reference SNVs in gnomAD (for Pejaver gft=0.03)
  cache/gnomad_myvariant.tsv         Cached MyVariant gnomAD v3.1.2 genome counts (af, grpmax)
  cohorts/cohort_summary.json        Exact counts by cohort, label, star level, and AF stratum
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import random
import re
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

CLINVAR_URLS = {
    "clinvar_20261108.vcf.gz": "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/archive_2.0/2026/clinvar_20261108.vcf.gz",
    "clinvar_20231226.vcf.gz": "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/archive_2.0/2023/clinvar_20231226.vcf.gz",
    "clinvar_20250330.vcf.gz": "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/archive_2.0/2025/clinvar_20250330.vcf.gz",
}

SEED = 20261123
ALLOWED_CHROMS = {str(i) for i in range(1, 23)} | {"X"}
BASES = {"A", "C", "G", "T"}
TWO_STAR_PLUS = {
    "criteria_provided,_multiple_submitters,_no_conflicts",
    "reviewed_by_expert_panel",
    "practice_guideline",
}
POS_SIGS = {"Pathogenic", "Likely_pathogenic", "Pathogenic/Likely_pathogenic"}
NEG_SIGS = {"Benign", "Likely_benign", "Benign/Likely_benign"}
CLASS_TOKENS = {"Pathogenic", "Likely_pathogenic", "Benign", "Likely_benign"}
SEVERE_CONSEQUENCES = (
    "nonsense",
    "stop_gained",
    "stop_lost",
    "initiator_codon",
    "frameshift",
    "splice_donor",
    "splice_acceptor",
)
GNOMAD_GROUPS = ["afr", "amr", "eas", "nfe", "sas"]
MAIN_AF = 0.001


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def download_vcf(url: str, dest: Path) -> dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists() or dest.stat().st_size == 0:
        tmp = dest.with_suffix(".part")
        for attempt in range(5):
            try:
                with urllib.request.urlopen(url, timeout=180) as r, tmp.open("wb") as out:
                    while chunk := r.read(1 << 20):
                        out.write(chunk)
                tmp.rename(dest)
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(5 * (attempt + 1))
    return {
        "filename": dest.name,
        "url": url,
        "bytes": dest.stat().st_size,
        "sha256": sha256_file(dest),
    }


def parse_info(info_str: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in info_str.split(";"):
        if "=" in item:
            k, v = item.split("=", 1)
            out[k] = v
    return out


def normalize_chrom(chrom: str) -> str | None:
    raw = chrom.removeprefix("chr")
    return f"chr{raw}" if raw in ALLOWED_CHROMS else None


def is_clean_missense(mc: str) -> bool:
    if "SO:0001583|missense_variant" not in mc:
        return False
    lower = mc.lower()
    return not any(tok in lower for tok in SEVERE_CONSEQUENCES)


def has_prior_pb_classification(clnsig: str) -> bool:
    if not clnsig:
        return False
    tokens = re.split(r"[,/|]+", clnsig)
    return any(tok in CLASS_TOKENS for tok in tokens)


def scan_archive_index(vcf_path: Path) -> tuple[set[tuple], set[str], set[tuple], set[str]]:
    """Return prior (classified_keys, classified_ids, all_seen_keys, all_seen_ids) from an earlier ClinVar VCF."""
    cls_keys: set[tuple] = set()
    cls_ids: set[str] = set()
    all_keys: set[tuple] = set()
    all_ids: set[str] = set()
    with gzip.open(vcf_path, "rt", errors="replace") as h:
        for line in h:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 8:
                continue
            chrom = normalize_chrom(fields[0])
            if chrom is None or fields[3] not in BASES or fields[4] not in BASES or fields[3] == fields[4]:
                continue
            key = (chrom, int(fields[1]), fields[3], fields[4])
            vid = fields[2]
            all_keys.add(key)
            if vid and vid != ".":
                all_ids.add(vid)
            info = parse_info(fields[7])
            if has_prior_pb_classification(info.get("CLNSIG", "")):
                cls_keys.add(key)
                if vid and vid != ".":
                    cls_ids.add(vid)
    return cls_keys, cls_ids, all_keys, all_ids


def parse_latest_clinvar(vcf_path: Path) -> tuple[list[dict], list[dict]]:
    """Extract >=2-star P/LP & B/LB missense SNVs and candidate gnomAD-observed VUS missense SNVs."""
    rows: list[dict] = []
    vus_candidates: list[dict] = []
    seen_keys: set[tuple] = set()
    with gzip.open(vcf_path, "rt", errors="replace") as h:
        for line in h:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 8:
                continue
            chrom = normalize_chrom(fields[0])
            if chrom is None or fields[3] not in BASES or fields[4] not in BASES or fields[3] == fields[4]:
                continue
            pos = int(fields[1])
            ref, alt = fields[3], fields[4]
            key = (chrom, pos, ref, alt)
            if key in seen_keys:
                continue
            info = parse_info(fields[7])
            mc = info.get("MC", "")
            if not is_clean_missense(mc):
                continue
            clnsig = info.get("CLNSIG", "")
            revstat = info.get("CLNREVSTAT", "")
            gene = info.get("GENEINFO", "").split("|")[0].split(":")[0]
            vid = fields[2]
            if revstat in TWO_STAR_PLUS and not info.get("CLNSIGCONF"):
                if clnsig in POS_SIGS:
                    label = 1
                elif clnsig in NEG_SIGS:
                    label = 0
                else:
                    label = -1
                if label in (0, 1):
                    seen_keys.add(key)
                    rows.append({
                        "chrom": chrom,
                        "pos": pos,
                        "ref": ref,
                        "alt": alt,
                        "variation_id": vid,
                        "label": label,
                        "clnsig": clnsig,
                        "clnrevstat": revstat,
                        "gene": gene,
                    })
                    continue
            if clnsig == "Uncertain_significance":
                has_pop_af = any(
                    float(info[k]) > 0
                    for k in ("AF_EXAC", "AF_TGP", "AF_ESP")
                    if k in info and info[k] not in ("", ".")
                )
                if has_pop_af:
                    seen_keys.add(key)
                    vus_candidates.append({
                        "chrom": chrom,
                        "pos": pos,
                        "ref": ref,
                        "alt": alt,
                        "source": "clinvar_vus_pop_af",
                    })
    rows.sort(key=lambda r: (r["chrom"], r["pos"], r["ref"], r["alt"]))
    return rows, vus_candidates


def query_myvariant_gnomad(keys: list[tuple], cache_path: Path, batch: int = 1000) -> dict[tuple, dict]:
    """Query gnomAD v3.1.2 genome counts from MyVariant.info with incremental TSV caching."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached: dict[tuple, dict] = {}
    if cache_path.exists():
        with cache_path.open() as h:
            for r in csv.DictReader(h, delimiter="\t"):
                k = (r["chrom"], int(r["pos"]), r["ref"], r["alt"])
                cached[k] = {
                    "in_gnomad": r["in_gnomad"] == "1",
                    "ac": int(r["ac"]),
                    "an": int(r["an"]),
                    "af": float(r["af"]),
                    "grpmax": float(r["grpmax"]),
                }
    todo = [k for k in sorted(set(keys)) if k not in cached]
    if todo:
        url = "https://myvariant.info/v1/variant?assembly=hg38&fields=gnomad_genome.ac,gnomad_genome.an"
        by_id = {f"{c}:g.{p}{r}>{a}": (c, p, r, a) for c, p, r, a in todo}
        ids = list(by_id)
        for i in range(0, len(ids), batch):
            chunk = ids[i : i + batch]
            body = urllib.parse.urlencode({"ids": ",".join(chunk)}).encode()
            for attempt in range(6):
                try:
                    req = urllib.request.Request(url, data=body)
                    with urllib.request.urlopen(req, timeout=180) as resp:
                        hits = json.load(resp)
                    break
                except OSError:
                    if attempt == 5:
                        raise
                    time.sleep(5 * (attempt + 1))
            seen_batch: set[tuple] = set()
            for h in hits:
                qkey = by_id.get(h["query"])
                if qkey is None or qkey in seen_batch:
                    continue
                seen_batch.add(qkey)
                g = h.get("gnomad_genome")
                if not g or h.get("notfound"):
                    cached[qkey] = {"in_gnomad": False, "ac": 0, "an": 0, "af": 0.0, "grpmax": 0.0}
                    continue
                if isinstance(g, list):
                    g = g[0]
                ac_obj = g.get("ac") or {}
                an_obj = g.get("an") or {}
                if isinstance(ac_obj, list):
                    ac_obj = ac_obj[0]
                if isinstance(an_obj, list):
                    an_obj = an_obj[0]
                ac_tot = int(ac_obj.get("ac", 0))
                an_tot = int(an_obj.get("an", 0))
                af = ac_tot / an_tot if an_tot else 0.0
                grpmax = max(
                    (int(ac_obj.get(f"ac_{grp}", 0)) / int(an_obj.get(f"an_{grp}", 0))
                     if int(an_obj.get(f"an_{grp}", 0)) else 0.0)
                    for grp in GNOMAD_GROUPS
                )
                cached[qkey] = {"in_gnomad": ac_tot > 0, "ac": ac_tot, "an": an_tot, "af": af, "grpmax": grpmax}
            for qid in chunk:
                qkey = by_id[qid]
                cached.setdefault(qkey, {"in_gnomad": False, "ac": 0, "an": 0, "af": 0.0, "grpmax": 0.0})
            if ((i // batch) + 1) % 5 == 0 or i + batch >= len(ids):
                print(f"  MyVariant gnomAD: {min(i + batch, len(ids))}/{len(ids)} queried", flush=True)
        with cache_path.open("w", newline="") as h:
            w = csv.writer(h, delimiter="\t", lineterminator="\n")
            w.writerow(["chrom", "pos", "ref", "alt", "in_gnomad", "ac", "an", "af", "grpmax"])
            for k in sorted(cached):
                d = cached[k]
                w.writerow([*k, int(d["in_gnomad"]), d["ac"], d["an"], f"{d['af']:.8g}", f"{d['grpmax']:.8g}"])
    return cached


def build_gnomad_missense_ref(
    excluded_keys: set[tuple],
    vus_candidates: list[dict],
    target_n: int = 7000,
    af5_snv_dir: Path | None = None,
) -> list[dict]:
    """Assemble population missense reference SNVs (optional 1000G/gnomAD bg missense + ClinVar VUS with pop AF)."""
    refs: dict[tuple, dict] = {}
    if af5_snv_dir is not None:
        cases_path = af5_snv_dir / "cases.tsv"
        cadd_path = af5_snv_dir / "scores_cadd.tsv"
    else:
        cases_path = cadd_path = None
    if cases_path is not None and cadd_path is not None and cases_path.exists() and cadd_path.exists():
        bg: dict[tuple, float] = {}
        with cases_path.open() as h:
            for r in csv.DictReader(h, delimiter="\t"):
                if r["diagnostic"] == "0":
                    k = (r["chrom"], int(r["pos"]), r["ref"], r["alt"])
                    if k not in excluded_keys and r["ref"] in BASES and r["alt"] in BASES:
                        bg[k] = float(r.get("grpmax_no1kg") or r.get("grpmax") or 0.0)
        with cadd_path.open() as h:
            for r in csv.DictReader(h, delimiter="\t"):
                k = (r["chrom"], int(r["pos"]), r["ref"], r["alt"])
                if k in bg and r["cadd_consequence"] == "NON_SYNONYMOUS":
                    refs[k] = {
                        "chrom": k[0],
                        "pos": k[1],
                        "ref": k[2],
                        "alt": k[3],
                        "source": "rare_disease_v2_af5_bg",
                        "af": "",
                        "grpmax": bg[k],
                    }
    rng = random.Random(SEED)
    pool = [v for v in vus_candidates if (v["chrom"], v["pos"], v["ref"], v["alt"]) not in excluded_keys and (v["chrom"], v["pos"], v["ref"], v["alt"]) not in refs]
    rng.shuffle(pool)
    for v in pool:
        if len(refs) >= target_n:
            break
        k = (v["chrom"], v["pos"], v["ref"], v["alt"])
        refs[k] = v
    return sorted(refs.values(), key=lambda r: (r["chrom"], r["pos"], r["ref"], r["alt"]))


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, default=BASE_DIR)
    parser.add_argument("--amp-studies-dir", type=Path, default=None)
    parser.add_argument("--af5-snv-dir", type=Path, default=None)
    args = parser.parse_args()
    raw_dir = args.workdir / "raw"
    cohorts_dir = args.workdir / "cohorts"
    cache_dir = args.workdir / "cache"
    af5_snv_dir = args.af5_snv_dir
    if af5_snv_dir is None and "REWIRE_AF5_SNV_DIR" in os.environ:
        af5_snv_dir = Path(os.environ["REWIRE_AF5_SNV_DIR"])
    if af5_snv_dir is None:
        amp_dir = args.amp_studies_dir or (
            Path(os.environ["REWIRE_AMP_STUDIES_DIR"]) if os.environ.get("REWIRE_AMP_STUDIES_DIR") else None
        )
        if amp_dir is not None:
            af5_snv_dir = amp_dir / "rare-disease-v2/data/af5/snv"

    print("Downloading / checking ClinVar archive VCFs...", flush=True)
    manifest = [download_vcf(url, raw_dir / name) for name, url in CLINVAR_URLS.items()]
    (raw_dir / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print("Scanning prior ClinVar archives (2023-12-26 and 2025-03-30)...", flush=True)
    cls_k_23, cls_id_23, seen_k_23, seen_id_23 = scan_archive_index(raw_dir / "clinvar_20231226.vcf.gz")
    cls_k_25, cls_id_25, seen_k_25, seen_id_25 = scan_archive_index(raw_dir / "clinvar_20250330.vcf.gz")

    print("Parsing clinvar_20261108.vcf.gz >=2-star missense SNVs...", flush=True)
    latest_rows, vus_candidates = parse_latest_clinvar(raw_dir / "clinvar_20261108.vcf.gz")
    all_pb_keys = {(r["chrom"], r["pos"], r["ref"], r["alt"]) for r in latest_rows}

    post23: list[dict] = []
    post25: list[dict] = []
    for r in latest_rows:
        k = (r["chrom"], r["pos"], r["ref"], r["alt"])
        vid = r["variation_id"]
        if k not in cls_k_23 and vid not in cls_id_23:
            row23 = dict(r)
            row23["novel_in_clinvar_2023"] = int(k not in seen_k_23 and vid not in seen_id_23)
            post23.append(row23)
        if k not in cls_k_25 and vid not in cls_id_25:
            row25 = dict(r)
            row25["novel_in_clinvar_2025"] = int(k not in seen_k_25 and vid not in seen_id_25)
            post25.append(row25)

    print(f"Held-out counts before AF lookup: post_2023_12={len(post23)}, post_2025_03={len(post25)}", flush=True)
    ref_candidates = build_gnomad_missense_ref(
        all_pb_keys | cls_k_23, vus_candidates, target_n=7500, af5_snv_dir=af5_snv_dir
    )
    all_query_keys = [(r["chrom"], r["pos"], r["ref"], r["alt"]) for r in post23]
    all_query_keys += [(r["chrom"], r["pos"], r["ref"], r["alt"]) for r in post25]
    all_query_keys += [(r["chrom"], r["pos"], r["ref"], r["alt"]) for r in ref_candidates]

    print(f"Querying gnomAD v3.1.2 genome counts via MyVariant for {len(set(all_query_keys))} unique keys...", flush=True)
    gnomad = query_myvariant_gnomad(all_query_keys, cache_dir / "gnomad_myvariant.tsv")

    for rows in (post23, post25):
        for r in rows:
            g = gnomad[(r["chrom"], r["pos"], r["ref"], r["alt"])]
            r["in_gnomad"] = int(g["in_gnomad"])
            r["gnomad_af"] = g["af"]
            r["gnomad_grpmax"] = g["grpmax"]
            r["rare_af001"] = int(g["grpmax"] < MAIN_AF)

    gnomad_ref = []
    for r in ref_candidates:
        g = gnomad[(r["chrom"], r["pos"], r["ref"], r["alt"])]
        if g["in_gnomad"] or r["source"] == "rare_disease_v2_af5_bg":
            gnomad_ref.append({
                "chrom": r["chrom"],
                "pos": r["pos"],
                "ref": r["ref"],
                "alt": r["alt"],
                "source": r["source"],
                "gnomad_af": g["af"],
                "gnomad_grpmax": g["grpmax"] if g["in_gnomad"] else r["grpmax"],
            })

    cols23 = ["chrom", "pos", "ref", "alt", "variation_id", "label", "clnsig", "clnrevstat", "gene",
              "novel_in_clinvar_2023", "in_gnomad", "gnomad_af", "gnomad_grpmax", "rare_af001"]
    cols25 = ["chrom", "pos", "ref", "alt", "variation_id", "label", "clnsig", "clnrevstat", "gene",
              "novel_in_clinvar_2025", "in_gnomad", "gnomad_af", "gnomad_grpmax", "rare_af001"]
    ref_cols = ["chrom", "pos", "ref", "alt", "source", "gnomad_af", "gnomad_grpmax"]

    write_tsv(cohorts_dir / "variants_post_2023_12.tsv", cols23, post23)
    write_tsv(cohorts_dir / "variants_post_2025_03.tsv", cols25, post25)
    write_tsv(cohorts_dir / "gnomad_ref_missense.tsv", ref_cols, gnomad_ref)

    summary = {
        "latest_all_2star_missense": dict(Counter(r["label"] for r in latest_rows)),
        "post_2023_12": {
            "total": len(post23),
            "pathogenic_lp": sum(r["label"] == 1 for r in post23),
            "benign_lb": sum(r["label"] == 0 for r in post23),
            "novel_total": sum(r["novel_in_clinvar_2023"] == 1 for r in post23),
            "rare_af001_pathogenic_lp": sum(r["label"] == 1 and r["rare_af001"] == 1 for r in post23),
            "rare_af001_benign_lb": sum(r["label"] == 0 and r["rare_af001"] == 1 for r in post23),
        },
        "post_2025_03": {
            "total": len(post25),
            "pathogenic_lp": sum(r["label"] == 1 for r in post25),
            "benign_lb": sum(r["label"] == 0 for r in post25),
            "novel_total": sum(r["novel_in_clinvar_2025"] == 1 for r in post25),
            "rare_af001_pathogenic_lp": sum(r["label"] == 1 and r["rare_af001"] == 1 for r in post25),
            "rare_af001_benign_lb": sum(r["label"] == 0 and r["rare_af001"] == 1 for r in post25),
        },
        "gnomad_ref_missense": {
            "total": len(gnomad_ref),
            "sources": dict(Counter(r["source"] for r in gnomad_ref)),
        },
    }
    (cohorts_dir / "cohort_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
