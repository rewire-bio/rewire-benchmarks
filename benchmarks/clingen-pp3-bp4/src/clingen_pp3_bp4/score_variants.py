"""Look up precomputed foundation model and baseline scores for ClinGen calibration variants.

Sources (local caches via CLI flags/env vars + public remote indexes, CPU only):
  am         AlphaMissense hg38 local TSV.GZ (--am-gz / REWIRE_AM_GZ)
  revel      REVEL v1.3 local zip (--revel-zip / REWIRE_REVEL_ZIP)
  avi        AlphaGenome Variant Impact local tabix (--avi-gz / REWIRE_AVI_GZ)
  gpnmsa     GPN-MSA hg38 remote tabix with local .tbi (-LLR)
  gpnstar_v  GPN-Star vertebrate v100 Parquet via DuckDB httpfs (-llr_calibrated)
  gpnstar_m  GPN-Star mammal m447 Parquet via DuckDB httpfs (-llr_calibrated)
  cadd       CADD v1.7 remote tabix with local .tbi (cadd_phred)
  evee       Optional local Parquet join (Zenodo 19701997; ClinVar-supervised through 2026-03-20, descriptive only)
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import os
import re
import subprocess
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
GPN_MSA_URL = "https://huggingface.co/datasets/songlab/gpn-msa-hg38-scores/resolve/main/scores.tsv.bgz"
GPN_STAR_URL = "hf://datasets/songlab/gpn-star-scores/data/gpn-star-hg38-{model}-200m/llr/llr_{chrom}.parquet"
CADD_URL = "https://kircherlab.bihealth.org/download/CADD/v1.7/GRCh38/whole_genome_SNVs.tsv.gz"
DEFAULT_CADD_TBI = BASE_DIR / "cache/whole_genome_SNVs_noanno.tsv.gz.tbi"


@dataclass
class SourcePaths:
    data_root: Path | None = None
    amp_studies_dir: Path | None = None
    am_gz: Path | None = None
    revel_zip: Path | None = None
    avi_gz: Path | None = None
    avi_tbi: Path | None = None
    gpn_msa_tbi: Path | None = None
    cadd_tbi: Path = DEFAULT_CADD_TBI
    evee_slim: Path | None = None

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "SourcePaths":
        data_root = args.data_root or (Path(os.environ["REWIRE_DATA_ROOT"]) if os.environ.get("REWIRE_DATA_ROOT") else None)
        amp_dir = args.amp_studies_dir or (
            Path(os.environ["REWIRE_AMP_STUDIES_DIR"])
            if os.environ.get("REWIRE_AMP_STUDIES_DIR")
            else (data_root / "amp-2026-studies" if data_root else None)
        )
        v1 = amp_dir / "rare-disease/data" if amp_dir else None
        v2 = amp_dir / "rare-disease-v2/data" if amp_dir else None
        am_gz = args.am_gz or (Path(os.environ["REWIRE_AM_GZ"]) if os.environ.get("REWIRE_AM_GZ") else (v1 / "am/AlphaMissense_hg38.tsv.gz" if v1 else None))
        revel_zip = args.revel_zip or (
            Path(os.environ["REWIRE_REVEL_ZIP"])
            if os.environ.get("REWIRE_REVEL_ZIP")
            else (data_root / "evo2-cloud-operations/data/raw/current-predictors/revel-v1.3_all_chromosomes.zip" if data_root else None)
        )
        avi_gz = args.avi_gz or (Path(os.environ["REWIRE_AVI_GZ"]) if os.environ.get("REWIRE_AVI_GZ") else None)
        avi_tbi = Path(f"{avi_gz}.tbi") if avi_gz else None
        gpn_msa_tbi = args.gpn_msa_tbi or (
            Path(os.environ["REWIRE_GPN_MSA_TBI"]) if os.environ.get("REWIRE_GPN_MSA_TBI") else (v1 / "gpn_msa.tsv.bgz.tbi" if v1 else None)
        )
        cadd_tbi = args.cadd_tbi or (Path(os.environ["REWIRE_CADD_TBI"]) if os.environ.get("REWIRE_CADD_TBI") else DEFAULT_CADD_TBI)
        evee_slim = args.evee_slim or (Path(os.environ["REWIRE_EVEE_SLIM"]) if os.environ.get("REWIRE_EVEE_SLIM") else (v2 / "evee/evee_slim.parquet" if v2 else None))
        return cls(
            data_root=data_root,
            amp_studies_dir=amp_dir,
            am_gz=am_gz,
            revel_zip=revel_zip,
            avi_gz=avi_gz,
            avi_tbi=avi_tbi,
            gpn_msa_tbi=gpn_msa_tbi,
            cadd_tbi=cadd_tbi,
            evee_slim=evee_slim,
        )

SEVERITY = {c: i for i, c in enumerate([
    "STOP_GAINED", "STOP_LOST", "CANONICAL_SPLICE", "NON_SYNONYMOUS", "SPLICE_SITE",
    "SYNONYMOUS", "5PRIME_UTR", "3PRIME_UTR", "NONCODING_CHANGE", "INTRONIC",
    "UPSTREAM", "DOWNSTREAM", "REGULATORY", "INTERGENIC",
])}

Key = tuple[str, int, str, str]


def load_keys(paths: list[Path]) -> set[Key]:
    keys: set[Key] = set()
    for p in paths:
        if not p.exists():
            continue
        with p.open() as h:
            for r in csv.DictReader(h, delimiter="\t"):
                keys.add((r["chrom"], int(r["pos"]), r["ref"], r["alt"]))
    return keys


def read_cached_scores(path: Path, header: list[str]) -> dict[Key, list[str]]:
    out: dict[Key, list[str]] = {}
    if not path.exists():
        return out
    with path.open() as h:
        for r in csv.DictReader(h, delimiter="\t"):
            out[(r["chrom"], int(r["pos"]), r["ref"], r["alt"])] = [r[c] for c in header]
    return out


def reuse_af5_bg(source_name: str, header: list[str], needed: set[Key], paths: SourcePaths) -> dict[Key, list[str]]:
    """Reuse optional rare-disease-v1/v2 scores for exact matching variant keys."""
    if not paths.amp_studies_dir:
        return {}
    v1_data = paths.amp_studies_dir / "rare-disease/data"
    v2_data = paths.amp_studies_dir / "rare-disease-v2/data"
    out: dict[Key, list[str]] = {}
    for directory in (v2_data / "af5/snv", v2_data / "snv", v2_data / "replication", v2_data / "pool", v1_data, v1_data / "replication"):
        candidate = directory / f"scores_{source_name}.tsv"
        if not candidate.exists():
            continue
        with candidate.open() as h:
            reader = csv.DictReader(h, delimiter="\t")
            if not set(header).issubset(set(reader.fieldnames or [])):
                continue
            for r in reader:
                k = (r["chrom"], int(r["pos"]), r["ref"], r["alt"])
                if k in needed:
                    out.setdefault(k, [r[c] for c in header])
    return out


def write_scores(path: Path, header: list[str], rows: dict[Key, list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as h:
        w = csv.writer(h, delimiter="\t", lineterminator="\n")
        w.writerow(["chrom", "pos", "ref", "alt", *header])
        for k in sorted(rows):
            w.writerow([*k, *rows[k]])
    tmp.rename(path)


def run_tabix(args: list[str], cwd: Path, label: str, tries: int = 8) -> str:
    for attempt in range(tries):
        r = subprocess.run(["tabix", *args], capture_output=True, text=True, cwd=cwd)
        if r.returncode == 0:
            return r.stdout
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"tabix failed ({label}): {r.stderr[-300:]}")


def lookup_am(wanted: set[Key], out_dir: Path, paths: SourcePaths) -> None:
    header = ["alphamissense"]
    out = out_dir / "scores_am.tsv"
    rows = read_cached_scores(out, header)
    rows.update({k: v for k, v in reuse_af5_bg("am", header, wanted - set(rows), paths).items()})
    todo = wanted - set(rows)
    if todo:
        assert paths.am_gz and paths.am_gz.exists(), "Pass --am-gz or REWIRE_AM_GZ to score uncached AlphaMissense variants"
        found: dict[Key, float] = {}
        with gzip.open(paths.am_gz, "rt") as h:
            for line in h:
                if line.startswith("#"):
                    continue
                f = line.rstrip("\n").split("\t")
                k = (f[0], int(f[1]), f[2], f[3])
                if k in todo:
                    val = float(f[8])
                    found[k] = max(val, found.get(k, val))
        rows.update({k: [f"{v:.6g}"] for k, v in found.items()})
    write_scores(out, header, rows)
    print(f"am: {len(rows)}/{len(wanted)} scored", flush=True)


def lookup_revel(wanted: set[Key], out_dir: Path, paths: SourcePaths) -> None:
    header = ["revel"]
    out = out_dir / "scores_revel.tsv"
    rows = read_cached_scores(out, header)
    rows.update({k: v for k, v in reuse_af5_bg("revel", header, wanted - set(rows), paths).items()})
    todo = wanted - set(rows)
    if todo:
        assert paths.revel_zip and paths.revel_zip.exists(), "Pass --revel-zip or REWIRE_REVEL_ZIP to score uncached REVEL variants"
        found: dict[Key, float] = {}
        with zipfile.ZipFile(paths.revel_zip) as z, z.open("revel_with_transcript_ids") as raw:
            for line in io.TextIOWrapper(raw):
                f = line.split(",")
                if f[2] == "." or not f[2].isdigit():
                    continue
                k = ("chr" + f[0], int(f[2]), f[3], f[4])
                if k in todo:
                    val = float(f[7])
                    found[k] = max(val, found.get(k, val))
        rows.update({k: [f"{v:.6g}"] for k, v in found.items()})
    write_scores(out, header, rows)
    print(f"revel: {len(rows)}/{len(wanted)} scored", flush=True)


def lookup_avi(wanted: set[Key], out_dir: Path, paths: SourcePaths) -> None:
    header = ["avi", "avi_phred"]
    out = out_dir / "scores_avi.tsv"
    rows = read_cached_scores(out, header)
    rows.update({k: v for k, v in reuse_af5_bg("avi", header, wanted - set(rows), paths).items()})
    todo = wanted - set(rows)
    if todo:
        assert paths.avi_gz and paths.avi_tbi and paths.avi_gz.exists() and paths.avi_tbi.exists(), (
            "Pass --avi-gz / REWIRE_AVI_GZ (with .tbi) to score uncached AVI variants"
        )
        with tempfile.TemporaryDirectory(prefix="avi_") as tmp:
            work = Path(tmp)
            chroms = sorted({c for c, *_ in todo})

            def one_chrom(chrom: str) -> dict[Key, list[str]]:
                pos = sorted({p for c, p, _, _ in todo if c == chrom})
                reg = work / f"reg_{chrom}.tsv"
                reg.write_text("".join(f"{chrom}\t{p}\t{p}\n" for p in pos))
                out_txt = run_tabix(["-R", str(reg), str(paths.avi_gz)], work, f"AVI {chrom}")
                res: dict[Key, list[str]] = {}
                for line in out_txt.splitlines():
                    f = line.split("\t")
                    k = (f[0], int(f[1]), f[2], f[3])
                    if k in todo:
                        res[k] = [f[4], f[5]]
                return res

            with ThreadPoolExecutor(4) as pool:
                for res in pool.map(one_chrom, chroms):
                    rows.update(res)
    write_scores(out, header, rows)
    print(f"avi: {len(rows)}/{len(wanted)} scored", flush=True)


def lookup_gpnmsa(wanted: set[Key], out_dir: Path, paths: SourcePaths) -> None:
    from concurrent.futures import as_completed

    header = ["gpn_msa"]
    out = out_dir / "scores_gpnmsa.tsv"
    rows = read_cached_scores(out, header)
    rows.update({k: v for k, v in reuse_af5_bg("gpnmsa", header, wanted - set(rows), paths).items()})
    todo = wanted - set(rows)
    if todo:
        assert paths.gpn_msa_tbi and paths.gpn_msa_tbi.exists(), (
            "Pass --gpn-msa-tbi or REWIRE_GPN_MSA_TBI to score uncached GPN-MSA variants"
        )
        with tempfile.TemporaryDirectory(prefix="gpnmsa_") as tmp:
            work = Path(tmp)
            tasks = []
            for chrom in sorted({c for c, *_ in todo}):
                pos = sorted({p for c, p, _, _ in todo if c == chrom})
                for idx in range(0, len(pos), 150):
                    tasks.append((chrom, idx, pos[idx : idx + 150]))

            def run_chunk(item: tuple[str, int, list[int]]) -> dict[Key, list[str]]:
                chrom, idx, chunk_pos = item
                reg = work / f"reg_{chrom}_{idx}.tsv"
                reg.write_text("".join(f"{chrom[3:]}\t{p}\t{p}\n" for p in chunk_pos))
                out_txt = run_tabix(["-R", str(reg), f"{GPN_MSA_URL}##idx##{paths.gpn_msa_tbi}"], work, f"GPN-MSA {chrom}")
                res: dict[Key, list[str]] = {}
                for line in out_txt.splitlines():
                    f = line.split("\t")
                    k = ("chr" + f[0], int(f[1]), f[2], f[3])
                    if k in todo:
                        res[k] = [f"{-float(f[4]):.6g}"]
                return res

            done = 0
            with ThreadPoolExecutor(12) as pool:
                futs = [pool.submit(run_chunk, t) for t in tasks]
                for fut in as_completed(futs):
                    rows.update(fut.result())
                    done += 1
                    if done % 10 == 0 or done == len(tasks):
                        write_scores(out, header, rows)
                        print(f"  gpnmsa chunk {done}/{len(tasks)} ({len(rows)}/{len(wanted)})", flush=True)
    write_scores(out, header, rows)
    print(f"gpnmsa: {len(rows)}/{len(wanted)} scored", flush=True)


def lookup_gpnstar(wanted: set[Key], out_dir: Path, paths: SourcePaths) -> None:
    from concurrent.futures import as_completed
    import threading
    import duckdb

    lock = threading.Lock()
    # Install httpfs once before spawning workers so concurrent duckdb threads don't race on extension install
    bootstrap_con = duckdb.connect()
    bootstrap_con.sql("INSTALL httpfs; LOAD httpfs;")
    bootstrap_con.close()

    for model, name in (("v100", "gpnstar_v"), ("m447", "gpnstar_m")):
        header = [name]
        out = out_dir / f"scores_{name}.tsv"
        chk = out_dir / f".{name}.done_chroms"
        done_chroms = set(chk.read_text().splitlines()) if chk.exists() else set()
        rows = read_cached_scores(out, header)
        rows.update({k: v for k, v in reuse_af5_bg(name, header, wanted - set(rows), paths).items()})
        todo = {k for k in (wanted - set(rows)) if k[0] not in done_chroms}
        chroms = sorted({c for c, *_ in todo})
        if chroms:
            def one_chrom(chrom: str, model: str = model, name: str = name) -> tuple[str, dict[Key, list[str]]]:
                con = duckdb.connect()
                con.sql("SET threads=2; LOAD httpfs;")
                positions = sorted({p for c, p, _, _ in todo if c == chrom})
                url = GPN_STAR_URL.format(model=model, chrom=chrom)
                found: dict[Key, list[str]] = {}
                for idx in range(0, len(positions), 500):
                    chunk = positions[idx : idx + 500]
                    q = (
                        f"SELECT pos, ref, alt, llr_calibrated FROM read_parquet('{url}') "
                        f"WHERE pos IN ({','.join(map(str, chunk))})"
                    )
                    for attempt in range(6):
                        try:
                            res = con.sql(q).fetchall()
                            break
                        except Exception as exc:
                            if attempt == 5:
                                raise RuntimeError(f"GPN-Star {model} {chrom} failed: {exc}") from exc
                            time.sleep(8 * (attempt + 1))
                    for pos, ref, alt, llr in res:
                        k = (chrom, int(pos), ref, alt)
                        if k in todo and llr is not None:
                            found[k] = [f"{-float(llr):.6g}"]
                con.close()
                return chrom, found

            with ThreadPoolExecutor(2) as pool:
                for fut in as_completed([pool.submit(one_chrom, c) for c in chroms]):
                    chrom, found = fut.result()
                    with lock:
                        rows.update(found)
                        done_chroms.add(chrom)
                        write_scores(out, header, rows)
                        chk.write_text("\n".join(sorted(done_chroms)) + "\n")
                        print(f"  {name} {chrom}: {len(rows)} cumulative ({len(done_chroms)} chroms done)", flush=True)
        write_scores(out, header, rows)
        print(f"{name}: {len(rows)}/{len(wanted)} scored", flush=True)


def mane_transcripts(mane_gtf: Path) -> set[str]:
    out: set[str] = set()
    with gzip.open(mane_gtf, "rt") as h:
        for line in h:
            if '\ttranscript\t' in line and 'tag "MANE_Select"' in line:
                m = re.search(r'transcript_id "([^".]+)', line)
                if m:
                    out.add(m.group(1))
    return out


def lookup_cadd(wanted: set[Key], out_dir: Path, paths: SourcePaths) -> None:
    from concurrent.futures import as_completed

    header = ["cadd_phred"]
    out = out_dir / "scores_cadd.tsv"
    rows = read_cached_scores(out, header)
    rows.update({k: v for k, v in reuse_af5_bg("cadd", header, wanted - set(rows), paths).items()})
    todo = wanted - set(rows)
    if todo:
        assert paths.cadd_tbi.exists(), f"Missing CADD index: {paths.cadd_tbi}"
        with tempfile.TemporaryDirectory(prefix="cadd_") as tmp:
            work = Path(tmp)
            tasks = []
            for chrom in sorted({c for c, *_ in todo}):
                pos = sorted({p for c, p, _, _ in todo if c == chrom})
                for idx in range(0, len(pos), 150):
                    tasks.append((chrom, idx, pos[idx : idx + 150]))

            def run_chunk(item: tuple[str, int, list[int]]) -> dict[Key, list[str]]:
                chrom, idx, chunk = item
                reg = work / f"cadd_{chrom}_{idx}.tsv"
                reg.write_text("".join(f"{chrom[3:]}\t{p}\t{p}\n" for p in chunk))
                txt = run_tabix(["-R", str(reg), f"{CADD_URL}##idx##{paths.cadd_tbi}"], work, f"CADD {chrom}")
                res: dict[Key, list[str]] = {}
                for line in txt.splitlines():
                    f = line.split("\t")
                    k = ("chr" + f[0], int(f[1]), f[2], f[3])
                    if k in todo:
                        res[k] = [f[5]]
                return res

            done = 0
            with ThreadPoolExecutor(12) as pool:
                futs = [pool.submit(run_chunk, t) for t in tasks]
                for fut in as_completed(futs):
                    rows.update(fut.result())
                    done += 1
                    if done % 10 == 0 or done == len(tasks):
                        write_scores(out, header, rows)
                        print(f"  cadd chunk {done}/{len(tasks)} ({len(rows)}/{len(wanted)})", flush=True)
    write_scores(out, header, rows)
    print(f"cadd: {len(rows)}/{len(wanted)} scored", flush=True)


def lookup_evee(wanted: set[Key], out_dir: Path, paths: SourcePaths) -> None:
    import duckdb

    header = ["evee", "evee_label", "evee_variation_id"]
    out = out_dir / "scores_evee.tsv"
    rows = read_cached_scores(out, header)
    todo = wanted - set(rows)
    if todo:
        assert paths.evee_slim and paths.evee_slim.exists(), (
            "Pass --evee-slim or REWIRE_EVEE_SLIM to score uncached EVEE variants"
        )
        con = duckdb.connect()
        con.execute("CREATE TABLE w (chrom VARCHAR, pos BIGINT, ref VARCHAR, alt VARCHAR)")
        con.executemany("INSERT INTO w VALUES (?, ?, ?, ?)", sorted(wanted))
        res = con.execute(f"""
            SELECT w.chrom, w.pos, w.ref, w.alt, e.pathogenicity, e.label, e.variation_id
            FROM w JOIN read_parquet('{paths.evee_slim}') e
              ON e.chrom = w.chrom AND e.pos = w.pos - 1 AND e.ref = w.ref AND e.alt = w.alt
        """).fetchall()
        rows = {
            (c, int(p), r, a): [
                "" if path is None else f"{float(path):.6g}",
                label or "",
                "" if vid is None else str(vid),
            ]
            for c, p, r, a, path, label, vid in res
        }
    write_scores(out, header, rows)
    print(f"evee (descriptive/in-sample caveat): {len(rows)}/{len(wanted)} matched", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workdir", type=Path, default=BASE_DIR)
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--amp-studies-dir", type=Path, default=None)
    p.add_argument("--am-gz", type=Path, default=None)
    p.add_argument("--revel-zip", type=Path, default=None)
    p.add_argument("--avi-gz", type=Path, default=None)
    p.add_argument("--gpn-msa-tbi", type=Path, default=None)
    p.add_argument("--cadd-tbi", type=Path, default=None)
    p.add_argument("--evee-slim", type=Path, default=None)
    p.add_argument(
        "sources", nargs="*",
        default=["am", "revel", "avi", "evee", "gpnstar", "gpnmsa", "cadd"],
    )
    args = p.parse_args()
    paths = SourcePaths.from_args(args)
    cohorts_dir = args.workdir / "cohorts"
    out_dir = args.workdir / "scores"
    wanted = load_keys([
        cohorts_dir / "variants_post_2023_12.tsv",
        cohorts_dir / "variants_post_2025_03.tsv",
        cohorts_dir / "gnomad_ref_missense.tsv",
    ])
    print(f"Scoring {len(wanted)} unique SNVs on CPU...", flush=True)
    runners = {
        "am": lookup_am,
        "revel": lookup_revel,
        "avi": lookup_avi,
        "evee": lookup_evee,
        "gpnmsa": lookup_gpnmsa,
        "gpnstar": lookup_gpnstar,
        "cadd": lookup_cadd,
    }
    for src in args.sources:
        t0 = time.time()
        runners[src](wanted, out_dir, paths)
        print(f"  completed {src} in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
