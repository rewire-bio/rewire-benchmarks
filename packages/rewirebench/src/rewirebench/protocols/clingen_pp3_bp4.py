"""ClinGen PP3/BP4 local posterior probability calibration protocol.

Implements Pejaver et al. 2022 (PMC9748256) and Bergquist et al. 2025 (PMC12208618)
Bayesian prior/odds thresholds, adaptive sliding windows with boundary scaling and
gnomAD reference density floors, and 1,000-resample one-sided conservative bounds.
"""

from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

PROTOCOL_ID = "clingen-pp3-bp4-v1"
PROTOCOL_VERSION = "1"
REVISION = "fbf4fd5df880ee685fa276dfa6a5bb46da4bcf73"

ALPHA = 0.0441
C_CONST = 1124.0
MINPOINTS = 100
GFT = 0.03
DEFAULT_SEED = 20261123
DEFAULT_N_BOOT = 1000
DEFAULT_STRATUM = "post_2023_12_af001"

GNOMAD_FILE = "gnomad_ref_missense.tsv"
GNOMAD_SHA256 = "818ad7178c61c6eeddb489d6969b98728895cd70baf95dcd3a3ec7008442ac83"

COHORT_2023_FILE = "variants_post_2023_12.tsv"
COHORT_2023_SHA256 = "90ec52d2b5dcbc4019113e60bd4b3099bf4e0761c821c2f2c7055b2c8b650438"
COHORT_2025_FILE = "variants_post_2025_03.tsv"
COHORT_2025_SHA256 = "0566277b1b393ee621144703e2c6e04f288a553d8bbc9fffb0553d9ac81b2156"

STRATA = {
    "post_2023_12": {
        "file": COHORT_2023_FILE,
        "sha256": COHORT_2023_SHA256,
        "rare_af001_only": False,
        "canonical_pos": 5508,
        "canonical_neg": 3942,
        "canonical_ref": 6312,
    },
    "post_2023_12_af001": {
        "file": COHORT_2023_FILE,
        "sha256": COHORT_2023_SHA256,
        "rare_af001_only": True,
        "canonical_pos": 5454,
        "canonical_neg": 2687,
        "canonical_ref": 6312,
    },
    "post_2025_03": {
        "file": COHORT_2025_FILE,
        "sha256": COHORT_2025_SHA256,
        "rare_af001_only": False,
        "canonical_pos": 2386,
        "canonical_neg": 1432,
        "canonical_ref": 6312,
    },
    "post_2025_03_af001": {
        "file": COHORT_2025_FILE,
        "sha256": COHORT_2025_SHA256,
        "rare_af001_only": True,
        "canonical_pos": 2368,
        "canonical_neg": 920,
        "canonical_ref": 6312,
    },
}

POINTS_LEVELS = (
    ("VeryStrong_8", 8),
    ("Strong_4", 4),
    ("Intermediate_3", 3),
    ("Moderate_2", 2),
    ("Supporting_1", 1),
)

CAPABILITIES = {
    "prediction_types": ["scalar"],
    "allow_fit": False,
    "allow_validation": False,
    "embedding_kind": None,
}


def describe() -> dict[str, Any]:
    return {
        "datasets": [f"clingen-pp3-bp4-{name}" for name in STRATA],
        "strata": list(STRATA),
        "default_stratum": DEFAULT_STRATUM,
        "splits": ["test"],
        "metrics": [
            "auroc",
            "auprc",
            "pp3_supporting",
            "pp3_moderate",
            "pp3_intermediate",
            "pp3_strong",
            "pp3_very_strong",
            "bp4_supporting",
            "bp4_moderate",
            "bp4_intermediate",
            "bp4_strong",
            "bp4_very_strong",
        ],
        "inputs": ["chrom", "pos", "ref", "alt"],
        "parameters": {
            "alpha": ALPHA,
            "c_const": C_CONST,
            "minpoints": MINPOINTS,
            "gft": GFT,
            "default_n_boot": DEFAULT_N_BOOT,
            "default_seed": DEFAULT_SEED,
        },
        "source_revision": REVISION,
        "data_access": (
            "Pinned cohort and gnomAD missense reference TSVs under "
            "benchmarks/clingen-pp3-bp4/cohorts in rewire-bio/rewire-benchmarks."
        ),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_input(source: Path, filename: str) -> Path:
    for candidate in (source / filename, source / "cohorts" / filename):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Missing required ClinGen input file: {filename} under {source}")


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def _make_row(row: dict[str, str], *, target: int, role: str) -> dict[str, Any]:
    chrom = row["chrom"]
    pos = int(row["pos"])
    ref = row["ref"]
    alt = row["alt"]
    return {
        "id": f"{chrom}:{pos}:{ref}:{alt}",
        "split": "test",
        "inputs": {"chrom": chrom, "pos": pos, "ref": ref, "alt": alt},
        "target": target,
        "role": role,
    }


def prepare(source: Path, **options):
    source = Path(source)
    if not source.is_dir():
        raise ValueError("source must be a directory holding ClinGen cohort TSVs")

    stratum = options.get("stratum", DEFAULT_STRATUM)
    if not isinstance(stratum, str) or stratum not in STRATA:
        raise ValueError(f"Unsupported ClinGen stratum {stratum!r}; choose {', '.join(STRATA)}")

    increment = options.get("increment", 0.001)
    if isinstance(increment, bool) or not isinstance(increment, (int, float)) or not math.isfinite(increment) or increment <= 0:
        raise ValueError("increment must be a positive finite number")

    decimals = options.get("decimals", 3)
    if isinstance(decimals, bool) or not isinstance(decimals, int) or decimals < 0 or decimals > 8:
        raise ValueError("decimals must be an integer between 0 and 8")

    n_boot = options.get("n_boot", DEFAULT_N_BOOT)
    if isinstance(n_boot, bool) or not isinstance(n_boot, int) or n_boot < 1:
        raise ValueError("n_boot must be a positive integer")

    seed = options.get("seed", DEFAULT_SEED)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    limit = options.get("limit")
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise ValueError("limit must be a positive integer")

    spec = STRATA[stratum]
    cohort_path = _resolve_input(source, spec["file"])
    gnomad_path = _resolve_input(source, GNOMAD_FILE)

    cohort_sha = _sha256_file(cohort_path)
    if cohort_sha != spec["sha256"]:
        raise ValueError(f"ClinGen cohort SHA-256 mismatch for {spec['file']}")
    gnomad_sha = _sha256_file(gnomad_path)
    if gnomad_sha != GNOMAD_SHA256:
        raise ValueError(f"ClinGen gnomAD reference SHA-256 mismatch for {GNOMAD_FILE}")

    cohort_raw = _read_tsv(cohort_path)
    if spec["rare_af001_only"]:
        cohort_raw = [r for r in cohort_raw if r.get("rare_af001") == "1"]
    gnomad_raw = _read_tsv(gnomad_path)

    pos_rows = [_make_row(r, target=1, role="pos") for r in cohort_raw if r.get("label") == "1"]
    neg_rows = [_make_row(r, target=0, role="neg") for r in cohort_raw if r.get("label") == "0"]
    ref_rows = [_make_row(r, target=-1, role="gnomad_ref") for r in gnomad_raw]

    if (len(pos_rows), len(neg_rows), len(ref_rows)) != (
        spec["canonical_pos"],
        spec["canonical_neg"],
        spec["canonical_ref"],
    ):
        raise ValueError(f"ClinGen stratum {stratum} variant counts changed")

    clinical_ids = {r["id"] for r in pos_rows} | {r["id"] for r in neg_rows}
    ref_ids = {r["id"] for r in ref_rows}
    if len(pos_rows) + len(neg_rows) != len(clinical_ids) or not clinical_ids.isdisjoint(ref_ids):
        raise ValueError("ClinGen cohort and gnomAD reference variants must be unique and disjoint")

    if limit is not None:
        pos_rows = pos_rows[:limit]
        neg_rows = neg_rows[:limit]
        ref_rows = ref_rows[:limit]

    rows = [*pos_rows, *neg_rows, *ref_rows]
    canonical_total = spec["canonical_pos"] + spec["canonical_neg"] + spec["canonical_ref"]
    return {
        "protocol_id": PROTOCOL_ID,
        "protocol_version": PROTOCOL_VERSION,
        "dataset_id": f"clingen-pp3-bp4-{stratum}",
        "scope": "smoke" if limit is not None else "full",
        "rows": rows,
        "provenance": {
            "cohort_file": spec["file"],
            "cohort_sha256": cohort_sha,
            "gnomad_file": GNOMAD_FILE,
            "gnomad_sha256": gnomad_sha,
            "upstream_revision": REVISION,
            "data_verification": "pinned_source_bytes",
            "source_url": f"https://github.com/rewire-bio/rewire-benchmarks/tree/{REVISION}/benchmarks/clingen-pp3-bp4",
        },
        "metadata": {
            "stratum": stratum,
            "increment": float(increment),
            "decimals": int(decimals),
            "n_boot": int(n_boot),
            "seed": int(seed),
            "canonical_pos_count": spec["canonical_pos"],
            "canonical_neg_count": spec["canonical_neg"],
            "canonical_ref_count": spec["canonical_ref"],
            "canonical_total_count": canonical_total,
            "smoke_limit_per_role": limit,
            "score_semantics": (
                "higher scalar score means stronger pathogenicity evidence; "
                "Pejaver 2022 and Bergquist 2025 local posterior odds calibration"
            ),
        },
    }


def validate_prepared(data: dict[str, Any]) -> None:
    if data.get("protocol_id") != PROTOCOL_ID or data.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Unsupported ClinGen PP3/BP4 protocol version")
    meta = data.get("metadata", {})
    if meta.get("stratum") not in STRATA:
        raise ValueError("Invalid ClinGen stratum in prepared metadata")
    seen_ids: set[str] = set()
    for row in data.get("rows", []):
        ident = row.get("id")
        role = row.get("role")
        inputs = row.get("inputs", {})
        if (
            not isinstance(ident, str)
            or ident in seen_ids
            or role not in {"pos", "neg", "gnomad_ref"}
            or set(inputs) != {"chrom", "pos", "ref", "alt"}
        ):
            raise ValueError("Malformed ClinGen prepared row")
        expected_target = 1 if role == "pos" else 0 if role == "neg" else -1
        if row.get("split") != "test" or row.get("target") != expected_target:
            raise ValueError("Malformed ClinGen row split or target")
        seen_ids.add(ident)


def posterior_cutoffs(
    alpha: float = ALPHA, c_const: float = C_CONST
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute Tavtigian/Pejaver pathogenic and benign posterior cutoffs."""
    post_p, post_b = {}, {}
    for name, pts in POINTS_LEVELS:
        lr = c_const ** (pts / 8.0)
        post_p[name] = (lr * alpha) / ((lr - 1.0) * alpha + 1.0)
        post_b[name] = (lr * (1.0 - alpha)) / ((lr - 1.0) * (1.0 - alpha) + 1.0)
    return post_p, post_b


def compute_local_posteriors_fast(
    pos_scores: np.ndarray,
    neg_scores: np.ndarray,
    g_scores: np.ndarray,
    thrs_desc: np.ndarray,
    increment: float,
    minpoints: int = MINPOINTS,
    gft: float = GFT,
    alpha: float = ALPHA,
) -> np.ndarray:
    """Vectorized binary-search Pejaver sliding window over descending grid."""
    pos_s = np.sort(pos_scores)
    neg_s = np.sort(neg_scores)
    g_s = np.sort(g_scores)
    maxthr = float(max(pos_s[-1], neg_s[-1], g_s[-1] if len(g_s) else neg_s[-1]))
    minthr = float(min(pos_s[0], neg_s[0], g_s[0] if len(g_s) else neg_s[0]))
    eps = 1e-12 * max(1.0, abs(maxthr - minthr))
    w = (1.0 - alpha) * len(pos_s) / (len(neg_s) * alpha)
    n_g_req = gft * len(g_s)
    k_max = int(math.ceil((maxthr - minthr) / increment)) + 2

    lo_k = np.zeros(len(thrs_desc), dtype=np.int64)
    hi_k = np.full(len(thrs_desc), k_max, dtype=np.int64)
    t = thrs_desc.astype(np.float64)

    while np.any(lo_k < hi_k):
        mid = (lo_k + hi_k) // 2
        half = mid * increment
        w_lo = t - half
        w_hi = t + half
        span = np.maximum(w_hi - w_lo, 0.0)
        c_bound = np.ones_like(t)
        clipped = (mid > 0) & ((w_lo < minthr) | (w_hi > maxthr))
        c_bound[clipped] = np.maximum(
            0.0,
            (np.minimum(w_hi[clipped], maxthr) - np.maximum(w_lo[clipped], minthr)) / span[clipped],
        )
        npos = np.searchsorted(pos_s, w_hi + eps, side="right") - np.searchsorted(
            pos_s, w_lo - eps, side="left"
        )
        nneg = np.searchsorted(neg_s, w_hi + eps, side="right") - np.searchsorted(
            neg_s, w_lo - eps, side="left"
        )
        if len(g_s):
            ng = np.searchsorted(g_s, w_hi + eps, side="right") - np.searchsorted(
                g_s, w_lo - eps, side="left"
            )
            ok_g = ng >= c_bound * n_g_req
        else:
            ok_g = True
        covered_all = (w_lo <= minthr - eps) & (w_hi >= maxthr + eps)
        ok = ((npos + nneg) >= c_bound * minpoints) & ok_g | covered_all
        hi_k = np.where(ok, mid, hi_k)
        lo_k = np.where(ok, lo_k, mid + 1)

    half = lo_k * increment
    w_lo = t - half
    w_hi = t + half
    npos = np.searchsorted(pos_s, w_hi + eps, side="right") - np.searchsorted(
        pos_s, w_lo - eps, side="left"
    )
    nneg = np.searchsorted(neg_s, w_hi + eps, side="right") - np.searchsorted(
        neg_s, w_lo - eps, side="left"
    )
    denom = npos + w * nneg
    return np.divide(
        npos.astype(np.float64),
        denom,
        out=np.full(denom.shape, alpha, dtype=np.float64),
        where=denom > 0,
    )


def extract_thresholds(
    thrs_desc: np.ndarray,
    post_p_desc: np.ndarray,
    post_p_cuts: dict[str, float],
    post_b_cuts: dict[str, float],
) -> tuple[dict[str, float], dict[str, float]]:
    """Translate local posterior curve to pathogenic (>=thr) and benign (<=thr) cutoffs."""
    p_out: dict[str, float] = {}
    for name, cutoff in post_p_cuts.items():
        below = np.flatnonzero(post_p_desc < cutoff)
        if len(below) == 0:
            p_out[name] = float(thrs_desc[-1])
        elif below[0] == 0:
            p_out[name] = math.nan
        else:
            p_out[name] = float(thrs_desc[below[0] - 1])

    thrs_asc = thrs_desc[::-1]
    post_b_asc = 1.0 - post_p_desc[::-1]
    b_out: dict[str, float] = {}
    for name, cutoff in post_b_cuts.items():
        below = np.flatnonzero(post_b_asc < cutoff)
        if len(below) == 0:
            b_out[name] = float(thrs_asc[-1])
        elif below[0] == 0:
            b_out[name] = math.nan
        else:
            b_out[name] = float(thrs_asc[below[0] - 1])
    return p_out, b_out


def discount_bootstrap_thresholds(values: np.ndarray, benign: bool = False) -> float:
    """Pejaver get_discounted_thresholds.m one-sided 95% conservative bound."""
    b_n = len(values)
    invalids = int(np.sum(np.isnan(values)))
    max_invalid = int(0.05 * b_n)
    if invalids > max_invalid:
        return math.nan
    valid = values[~np.isnan(values)]
    valid_sorted = np.sort(valid) if benign else np.sort(valid)[::-1]
    idx = max(0, int(math.floor(0.05 * b_n)) - invalids)
    return float(valid_sorted[min(idx, len(valid_sorted) - 1)])


def _finite_or_none(val: float) -> float | None:
    return None if math.isnan(val) or math.isinf(val) else float(val)


def calibrate(
    pos_scores: np.ndarray,
    neg_scores: np.ndarray,
    g_scores: np.ndarray,
    *,
    increment: float,
    decimals: int,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    """Run Pejaver point estimation and bootstrap threshold discounting."""
    post_p_cuts, post_b_cuts = posterior_cutoffs()
    all_vals = (
        np.concatenate([pos_scores, neg_scores, g_scores])
        if len(g_scores)
        else np.concatenate([pos_scores, neg_scores])
    )
    grid = np.unique(np.round(all_vals, decimals))
    if len(grid) > 2000:
        grid = grid[np.unique(np.linspace(0, len(grid) - 1, 2000, dtype=int))]
    thrs_desc = grid[::-1]

    post_p_point = compute_local_posteriors_fast(
        pos_scores, neg_scores, g_scores, thrs_desc, increment
    )
    p_point, b_point = extract_thresholds(thrs_desc, post_p_point, post_p_cuts, post_b_cuts)

    rng = np.random.Generator(np.random.PCG64(seed))
    n_p, n_n, n_g = len(pos_scores), len(neg_scores), len(g_scores)
    boot_p = {name: np.empty(n_boot, dtype=np.float64) for name, _ in POINTS_LEVELS}
    boot_b = {name: np.empty(n_boot, dtype=np.float64) for name, _ in POINTS_LEVELS}
    boot_auroc = np.empty(n_boot, dtype=np.float64)
    boot_auprc = np.empty(n_boot, dtype=np.float64)
    y_true_full = np.concatenate([np.ones(n_p, dtype=np.int8), np.zeros(n_n, dtype=np.int8)])
    scores_full = np.concatenate([pos_scores, neg_scores])

    for b in range(n_boot):
        idx_p = rng.integers(0, n_p, size=n_p)
        idx_n = rng.integers(0, n_n, size=n_n)
        ps = pos_scores[idx_p]
        ns = neg_scores[idx_n]
        gs = g_scores[rng.integers(0, n_g, size=n_g)] if n_g else g_scores
        pp_curve = compute_local_posteriors_fast(ps, ns, gs, thrs_desc, increment)
        pp_b, pb_b = extract_thresholds(thrs_desc, pp_curve, post_p_cuts, post_b_cuts)
        for name, _ in POINTS_LEVELS:
            boot_p[name][b] = pp_b[name]
            boot_b[name][b] = pb_b[name]
        yb = np.concatenate([np.ones(n_p, dtype=np.int8), np.zeros(n_n, dtype=np.int8)])
        sb = np.concatenate([ps, ns])
        boot_auroc[b] = roc_auc_score(yb, sb)
        boot_auprc[b] = average_precision_score(yb, sb)

    p_disc = {
        name: discount_bootstrap_thresholds(boot_p[name], benign=False)
        for name, _ in POINTS_LEVELS
    }
    b_disc = {
        name: discount_bootstrap_thresholds(boot_b[name], benign=True)
        for name, _ in POINTS_LEVELS
    }

    def assign_tier(vals: np.ndarray) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        assigned = np.zeros(len(vals), dtype=bool)
        for name, pts in POINTS_LEVELS:
            thr = p_disc[name]
            if not math.isnan(thr):
                hit = (~assigned) & (vals >= thr)
                assigned |= hit
                out[f"PP3_{name}"] = {"points": pts, "count": int(np.sum(hit)), "share": float(np.mean(hit))}
            else:
                out[f"PP3_{name}"] = {"points": pts, "count": 0, "share": 0.0}
        b_assigned = np.zeros(len(vals), dtype=bool)
        for name, pts in POINTS_LEVELS:
            thr = b_disc[name]
            if not math.isnan(thr) and (math.isnan(p_disc["Supporting_1"]) or thr < p_disc["Supporting_1"]):
                hit = (~assigned) & (~b_assigned) & (vals <= thr)
                b_assigned |= hit
                out[f"BP4_{name}"] = {"points": -pts, "count": int(np.sum(hit)), "share": float(np.mean(hit))}
            else:
                out[f"BP4_{name}"] = {"points": -pts, "count": 0, "share": 0.0}
        indet = (~assigned) & (~b_assigned)
        out["Indeterminate_0"] = {"points": 0, "count": int(np.sum(indet)), "share": float(np.mean(indet))}
        return out

    interval_lr: dict[str, dict[str, Any]] = {}
    prev_hi = math.inf
    for name, pts in POINTS_LEVELS:
        lo = p_disc[name]
        if math.isnan(lo) or lo >= prev_hi:
            interval_lr[f"PP3_{name}"] = {"points": pts, "n_pos": 0, "n_neg": 0, "lr_plus": None}
            continue
        p_c = int(np.sum((pos_scores >= lo) & (pos_scores < prev_hi)))
        n_c = int(np.sum((neg_scores >= lo) & (neg_scores < prev_hi)))
        lr = ((p_c / n_p) / (n_c / n_n)) if n_c > 0 else (math.inf if p_c > 0 else None)
        interval_lr[f"PP3_{name}"] = {
            "points": pts,
            "n_pos": p_c,
            "n_neg": n_c,
            "lr_plus": None if lr is None or math.isinf(lr) else float(lr),
        }
        prev_hi = lo

    return {
        "n_pos": n_p,
        "n_neg": n_n,
        "n_g_ref": n_g,
        "auroc": float(roc_auc_score(y_true_full, scores_full)),
        "auroc_ci95": [float(np.quantile(boot_auroc, 0.025)), float(np.quantile(boot_auroc, 0.975))],
        "auprc": float(average_precision_score(y_true_full, scores_full)),
        "auprc_ci95": [float(np.quantile(boot_auprc, 0.025)), float(np.quantile(boot_auprc, 0.975))],
        "point_thresholds": {
            "PP3": {k: _finite_or_none(v) for k, v in p_point.items()},
            "BP4": {k: _finite_or_none(v) for k, v in b_point.items()},
        },
        "discounted_thresholds": {
            "PP3": {k: _finite_or_none(v) for k, v in p_disc.items()},
            "BP4": {k: _finite_or_none(v) for k, v in b_disc.items()},
        },
        "pathogenic_reach": assign_tier(pos_scores),
        "benign_reach": assign_tier(neg_scores),
        "interval_lr": interval_lr,
    }


def score(dataset: dict[str, Any], predictions: dict[str, float]) -> dict[str, Any]:
    """Score keyed variant predictions with Pejaver/Bergquist ClinGen calibration."""
    validate_prepared(dataset)
    rows = dataset["rows"]
    known = {r["id"] for r in rows}
    if set(predictions) - known:
        raise ValueError("Unknown ClinGen prediction IDs")
    if not all(
        isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
        for v in predictions.values()
    ):
        raise ValueError("ClinGen predictions must be finite scalar scores")

    meta = dataset["metadata"]
    pos_scores = np.array(
        [predictions[r["id"]] for r in rows if r["role"] == "pos" and r["id"] in predictions],
        dtype=np.float64,
    )
    neg_scores = np.array(
        [predictions[r["id"]] for r in rows if r["role"] == "neg" and r["id"] in predictions],
        dtype=np.float64,
    )
    g_scores = np.array(
        [predictions[r["id"]] for r in rows if r["role"] == "gnomad_ref" and r["id"] in predictions],
        dtype=np.float64,
    )
    n_p, n_n, n_g = len(pos_scores), len(neg_scores), len(g_scores)
    denominator = int(meta["canonical_total_count"])

    if n_p < 2 or n_n < 2:
        metrics: dict[str, Any] = {
            "n": len(predictions),
            "n_pos": n_p,
            "n_neg": n_n,
            "n_ref": n_g,
            "auroc": None,
            "auprc": None,
        }
        return {
            "protocol_id": PROTOCOL_ID,
            "scope": dataset["scope"],
            "complete": False,
            "metrics": metrics,
            "coverage": {
                "denominator": denominator,
                "scored": len(predictions),
                "unscored": denominator - len(predictions),
            },
            "unavailable_reason": "At least two pathogenic and two benign scored variants required",
        }

    cal = calibrate(
        pos_scores,
        neg_scores,
        g_scores,
        increment=float(meta["increment"]),
        decimals=int(meta["decimals"]),
        n_boot=int(meta["n_boot"]),
        seed=int(meta["seed"]),
    )
    disc_p = cal["discounted_thresholds"]["PP3"]
    disc_b = cal["discounted_thresholds"]["BP4"]
    p_reach = cal["pathogenic_reach"]
    b_reach = cal["benign_reach"]
    plp_supp = sum(p_reach[f"PP3_{name}"]["share"] for name, _ in POINTS_LEVELS)
    plp_str = sum(p_reach[f"PP3_{name}"]["share"] for name, pts in POINTS_LEVELS if pts >= 4)
    blb_supp = sum(b_reach[f"BP4_{name}"]["share"] for name, _ in POINTS_LEVELS)
    blb_mod = sum(b_reach[f"BP4_{name}"]["share"] for name, pts in POINTS_LEVELS if pts >= 2)

    metrics = {
        "n": len(predictions),
        "n_pos": n_p,
        "n_neg": n_n,
        "n_ref": n_g,
        "auroc": cal["auroc"],
        "auroc_ci95_lo": cal["auroc_ci95"][0],
        "auroc_ci95_hi": cal["auroc_ci95"][1],
        "auprc": cal["auprc"],
        "auprc_ci95_lo": cal["auprc_ci95"][0],
        "auprc_ci95_hi": cal["auprc_ci95"][1],
        "pp3_supporting": disc_p["Supporting_1"],
        "pp3_moderate": disc_p["Moderate_2"],
        "pp3_intermediate": disc_p["Intermediate_3"],
        "pp3_strong": disc_p["Strong_4"],
        "pp3_very_strong": disc_p["VeryStrong_8"],
        "bp4_supporting": disc_b["Supporting_1"],
        "bp4_moderate": disc_b["Moderate_2"],
        "bp4_intermediate": disc_b["Intermediate_3"],
        "bp4_strong": disc_b["Strong_4"],
        "bp4_very_strong": disc_b["VeryStrong_8"],
        "plp_reaching_pp3_supporting_share": float(plp_supp),
        "plp_reaching_pp3_strong_share": float(plp_str),
        "blb_reaching_bp4_supporting_share": float(blb_supp),
        "blb_reaching_bp4_moderate_share": float(blb_mod),
        "max_pp3_points": max((pts for name, pts in POINTS_LEVELS if disc_p[name] is not None), default=0),
        "max_bp4_points": max((pts for name, pts in POINTS_LEVELS if disc_b[name] is not None), default=0),
        "coverage_pos": float(n_p / meta["canonical_pos_count"]),
        "coverage_neg": float(n_n / meta["canonical_neg_count"]),
        "coverage_ref": float(n_g / meta["canonical_ref_count"]),
    }

    return {
        "protocol_id": PROTOCOL_ID,
        "scope": dataset["scope"],
        "complete": dataset["scope"] == "full" and len(predictions) == denominator,
        "metrics": metrics,
        "stratum": meta["stratum"],
        "point_thresholds": cal["point_thresholds"],
        "discounted_thresholds": cal["discounted_thresholds"],
        "pathogenic_reach": p_reach,
        "benign_reach": b_reach,
        "interval_lr": cal["interval_lr"],
        "coverage": {
            "denominator": denominator,
            "scored": len(predictions),
            "unscored": denominator - len(predictions),
        },
        "score_direction": "higher means stronger pathogenicity evidence",
    }
