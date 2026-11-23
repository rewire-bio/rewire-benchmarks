"""ClinGen PP3/BP4 local posterior probability calibration (Pejaver et al. 2022, Bergquist et al. 2025).

Implements exact Pejaver et al. 2022 (PMC9748256) and Bergquist et al. 2025 (PMC12208618) Bayesian
prior/odds thresholds, adaptive sliding windows with boundary scaling and gnomAD density floors,
and 1,000-resample one-sided 95% conservative bootstrap intervals.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

BASE_DIR = Path(__file__).resolve().parents[2]
SEED = 20261123

# Pejaver et al. 2022 (PMC9748256, main.m): prior alpha=0.0441, c=1124 (O_PVS1=350)
ALPHA = 0.0441
C_CONST = 1124.0
MINPOINTS = 100
GFT = 0.03

# Points levels from Pejaver 2022 (+1,+2,+4,+8) and Bergquist 2025 (includes +-3 intermediate)
POINTS_LEVELS = (
    ("VeryStrong_8", 8),
    ("Strong_4", 4),
    ("Intermediate_3", 3),
    ("Moderate_2", 2),
    ("Supporting_1", 1),
)

BERGQUIST_2025_TABLE1 = {
    "revel": {
        "BP4_Strong_4": ("le", 0.016),
        "BP4_Intermediate_3": ("interval", 0.017, 0.052),
        "BP4_Moderate_2": ("interval", 0.053, 0.183),
        "BP4_Supporting_1": ("interval", 0.184, 0.290),
        "Indeterminate_0": ("interval", 0.291, 0.643),
        "PP3_Supporting_1": ("interval", 0.644, 0.772),
        "PP3_Moderate_2": ("interval", 0.773, 0.878),
        "PP3_Intermediate_3": ("interval", 0.879, 0.931),
        "PP3_Strong_4": ("ge", 0.932),
    },
    "alphamissense": {
        "BP4_Intermediate_3": ("le", 0.070),
        "BP4_Moderate_2": ("interval", 0.071, 0.099),
        "BP4_Supporting_1": ("interval", 0.100, 0.169),
        "Indeterminate_0": ("interval", 0.170, 0.791),
        "PP3_Supporting_1": ("interval", 0.792, 0.905),
        "PP3_Moderate_2": ("interval", 0.906, 0.971),
        "PP3_Intermediate_3": ("interval", 0.972, 0.989),
        "PP3_Strong_4": ("ge", 0.990),
    },
}


@dataclass(frozen=True)
class PredictorSpec:
    key: str
    label: str
    score_file: str
    column: str
    increment: float
    decimals: int
    role: str  # "foundation_model" | "clinical_baseline" | "descriptive_leaked"
    note: str = ""


PREDICTORS = [
    PredictorSpec("gpn_msa", "GPN-MSA (-LLR)", "scores_gpnmsa.tsv", "gpn_msa", 0.01, 2, "foundation_model"),
    PredictorSpec("gpnstar_v", "GPN-Star vertebrate v100 (-LLR)", "scores_gpnstar_v.tsv", "gpnstar_v", 0.01, 2, "foundation_model"),
    PredictorSpec("gpnstar_m", "GPN-Star mammal m447 (-LLR)", "scores_gpnstar_m.tsv", "gpnstar_m", 0.01, 2, "foundation_model"),
    PredictorSpec("avi", "AlphaGenome AVI (raw)", "scores_avi.tsv", "avi", 0.005, 3, "foundation_model",
                  "Trained on gnomAD allele frequency classes; inspect AF<0.1% sub-stratum."),
    PredictorSpec("avi_phred", "AlphaGenome AVI (PHRED)", "scores_avi.tsv", "avi_phred", 0.1, 1, "foundation_model",
                  "PHRED-scaled AVI; same gnomAD frequency training structure as raw AVI."),
    PredictorSpec("alphamissense", "AlphaMissense", "scores_am.tsv", "alphamissense", 0.001, 3, "clinical_baseline"),
    PredictorSpec("revel", "REVEL v1.3", "scores_revel.tsv", "revel", 0.001, 3, "clinical_baseline"),
    PredictorSpec("cadd_phred", "CADD v1.7 (PHRED)", "scores_cadd.tsv", "cadd_phred", 0.1, 1, "clinical_baseline"),
    PredictorSpec("evee", "EVEE (ClinVar-supervised, descriptive)", "scores_evee.tsv", "evee", 0.001, 3,
                  "descriptive_leaked", "Published table trained with ClinVar labels through 2026-03-20."),
]


def posterior_cutoffs(alpha: float = ALPHA, c_const: float = C_CONST) -> tuple[dict[str, float], dict[str, float]]:
    """Compute Tavtigian/Pejaver pathogenic & benign posterior cutoffs for 8, 4, 3, 2, 1 points."""
    post_p, post_b = {}, {}
    for name, pts in POINTS_LEVELS:
        lr = c_const ** (pts / 8.0)
        post_p[name] = (lr * alpha) / ((lr - 1.0) * alpha + 1.0)
        post_b[name] = (lr * (1.0 - alpha)) / ((lr - 1.0) * (1.0 - alpha) + 1.0)
    return post_p, post_b


def compute_local_posteriors_reference(
    pos_scores: np.ndarray,
    neg_scores: np.ndarray,
    g_scores: np.ndarray,
    thrs_desc: np.ndarray,
    increment: float,
    minpoints: int = MINPOINTS,
    gft: float = GFT,
    alpha: float = ALPHA,
) -> np.ndarray:
    """Direct translation of Pejaver et al. 2022 get_both_local_posteriors.m (for tests & parity)."""
    pos_s = np.sort(pos_scores)
    neg_s = np.sort(neg_scores)
    g_s = np.sort(g_scores)
    maxthr = float(np.max(np.concatenate([pos_s, neg_s, g_s]))) if len(g_s) else float(np.max(np.concatenate([pos_s, neg_s])))
    minthr = float(np.min(np.concatenate([pos_s, neg_s, g_s]))) if len(g_s) else float(np.min(np.concatenate([pos_s, neg_s])))
    w = (1.0 - alpha) * len(pos_s) / (len(neg_s) * alpha)
    n_g_req = gft * len(g_s)
    post_p = np.empty(len(thrs_desc), dtype=np.float64)
    eps = 1e-12 * max(1.0, abs(maxthr - minthr))
    for i, t in enumerate(thrs_desc):
        k = 0
        while True:
            half = k * increment
            lo = t - half
            hi = t + half
            if lo < minthr or hi > maxthr:
                span = hi - lo
                if span <= 0:
                    c_bound = 1.0
                else:
                    c_bound = max(0.0, (min(hi, maxthr) - max(lo, minthr)) / span)
            else:
                c_bound = 1.0
            npos = int(np.searchsorted(pos_s, hi + eps, side="right") - np.searchsorted(pos_s, lo - eps, side="left"))
            nneg = int(np.searchsorted(neg_s, hi + eps, side="right") - np.searchsorted(neg_s, lo - eps, side="left"))
            ng = int(np.searchsorted(g_s, hi + eps, side="right") - np.searchsorted(g_s, lo - eps, side="left")) if len(g_s) else 0
            enough_pn = (npos + nneg) >= c_bound * minpoints
            enough_g = (len(g_s) == 0) or (ng >= c_bound * n_g_req)
            if (enough_pn and enough_g) or (lo <= minthr - eps and hi >= maxthr + eps):
                denom = npos + w * nneg
                post_p[i] = npos / denom if denom > 0 else alpha
                break
            k += 1
    return post_p


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
    """Vectorized binary-search Pejaver sliding window over descending grid thrs_desc."""
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
        npos = np.searchsorted(pos_s, w_hi + eps, side="right") - np.searchsorted(pos_s, w_lo - eps, side="left")
        nneg = np.searchsorted(neg_s, w_hi + eps, side="right") - np.searchsorted(neg_s, w_lo - eps, side="left")
        if len(g_s):
            ng = np.searchsorted(g_s, w_hi + eps, side="right") - np.searchsorted(g_s, w_lo - eps, side="left")
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
    npos = np.searchsorted(pos_s, w_hi + eps, side="right") - np.searchsorted(pos_s, w_lo - eps, side="left")
    nneg = np.searchsorted(neg_s, w_hi + eps, side="right") - np.searchsorted(neg_s, w_lo - eps, side="left")
    denom = npos + w * nneg
    return np.divide(npos.astype(np.float64), denom, out=np.full(denom.shape, alpha, dtype=np.float64), where=denom > 0)


def extract_thresholds(
    thrs_desc: np.ndarray,
    post_p_desc: np.ndarray,
    post_p_cuts: dict[str, float],
    post_b_cuts: dict[str, float],
) -> tuple[dict[str, float], dict[str, float]]:
    """Translate local posterior curve to raw pathogenic (>=thr) and benign (<=thr) cutoffs."""
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


def calibrate_predictor(
    pos_scores: np.ndarray,
    neg_scores: np.ndarray,
    g_scores: np.ndarray,
    increment: float,
    decimals: int,
    n_boot: int = 1000,
    seed: int = SEED,
) -> dict[str, Any]:
    """Run Pejaver point estimation + n_boot one-sided conservative threshold calibration."""
    post_p_cuts, post_b_cuts = posterior_cutoffs()
    all_vals = np.concatenate([pos_scores, neg_scores, g_scores]) if len(g_scores) else np.concatenate([pos_scores, neg_scores])
    grid = np.unique(np.round(all_vals, decimals))
    if len(grid) > 2000:
        # Preserve empirical density where variants concentrate while capping grid size for 1,000 bootstraps
        grid = grid[np.unique(np.linspace(0, len(grid) - 1, 2000, dtype=int))]
    thrs_desc = grid[::-1]

    post_p_point = compute_local_posteriors_fast(pos_scores, neg_scores, g_scores, thrs_desc, increment)
    p_point, b_point = extract_thresholds(thrs_desc, post_p_point, post_p_cuts, post_b_cuts)

    # Bootstrap stratified over positive/negative classes + g_scores (if present)
    rng = np.random.default_rng(seed)
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

    p_disc = {name: discount_bootstrap_thresholds(boot_p[name], benign=False) for name, _ in POINTS_LEVELS}
    b_disc = {name: discount_bootstrap_thresholds(boot_b[name], benign=True) for name, _ in POINTS_LEVELS}

    # Share of held-out P/LP and B/LB reaching each tier using discounted thresholds
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

    # Post-hoc local LR+ inside each non-overlapping discounted tier
    def tier_likelihood_ratios() -> dict[str, dict[str, Any]]:
        res: dict[str, dict[str, Any]] = {}
        prev_hi = math.inf
        for name, pts in POINTS_LEVELS:
            lo = p_disc[name]
            if math.isnan(lo) or lo >= prev_hi:
                res[f"PP3_{name}"] = {"points": pts, "n_pos": 0, "n_neg": 0, "lr_plus": None}
                continue
            p_c = int(np.sum((pos_scores >= lo) & (pos_scores < prev_hi)))
            n_c = int(np.sum((neg_scores >= lo) & (neg_scores < prev_hi)))
            lr = ((p_c / n_p) / (n_c / n_n)) if n_c > 0 else (math.inf if p_c > 0 else None)
            res[f"PP3_{name}"] = {"points": pts, "n_pos": p_c, "n_neg": n_c, "lr_plus": None if lr is None or math.isinf(lr) else float(lr)}
            prev_hi = lo
        return res

    def _finite_or_none(v: float) -> float | None:
        return None if math.isnan(v) or math.isinf(v) else float(v)

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
        "bootstrap_intervals_95": {
            "PP3": {
                name: [_finite_or_none(float(np.nanquantile(boot_p[name], 0.025))), _finite_or_none(float(np.nanquantile(boot_p[name], 0.975)))]
                if int(np.sum(np.isnan(boot_p[name]))) <= int(0.05 * n_boot) else None
                for name, _ in POINTS_LEVELS
            },
            "BP4": {
                name: [_finite_or_none(float(np.nanquantile(boot_b[name], 0.025))), _finite_or_none(float(np.nanquantile(boot_b[name], 0.975)))]
                if int(np.sum(np.isnan(boot_b[name]))) <= int(0.05 * n_boot) else None
                for name, _ in POINTS_LEVELS
            },
        },
        "pathogenic_reach": assign_tier(pos_scores),
        "benign_reach": assign_tier(neg_scores),
        "interval_lr": tier_likelihood_ratios(),
    }


def evaluate_bergquist_table1(predictor_key: str, pos_scores: np.ndarray, neg_scores: np.ndarray) -> dict[str, dict[str, Any]]:
    """Compute post-hoc empirical LR+ inside Bergquist et al. 2025 (PMC12208618) Table 1 intervals."""
    spec = BERGQUIST_2025_TABLE1[predictor_key]
    n_p, n_n = len(pos_scores), len(neg_scores)
    out: dict[str, dict[str, Any]] = {}
    for tier, rule in spec.items():
        kind = rule[0]
        if kind == "le":
            mask_p, mask_n = pos_scores <= rule[1], neg_scores <= rule[1]
        elif kind == "ge":
            mask_p, mask_n = pos_scores >= rule[1], neg_scores >= rule[1]
        else:
            mask_p = (pos_scores >= rule[1]) & (pos_scores <= rule[2])
            mask_n = (neg_scores >= rule[1]) & (neg_scores <= rule[2])
        p_c, n_c = int(np.sum(mask_p)), int(np.sum(mask_n))
        lr = ((p_c / n_p) / (n_c / n_n)) if n_c > 0 else (math.inf if p_c > 0 else None)
        out[tier] = {
            "n_pos": p_c,
            "n_neg": n_c,
            "lr_plus": None if lr is None or math.isinf(lr) else float(lr),
            "pos_share": float(p_c / n_p) if n_p else 0.0,
            "neg_share": float(n_c / n_n) if n_n else 0.0,
        }
    return out


def load_cohort(path: Path) -> list[dict]:
    with path.open() as h:
        return list(csv.DictReader(h, delimiter="\t"))


def load_score_map(path: Path, col: str) -> dict[tuple[str, int, str, str], float]:
    out: dict[tuple[str, int, str, str], float] = {}
    if not path.exists():
        return out
    with path.open() as h:
        for r in csv.DictReader(h, delimiter="\t"):
            v = r.get(col, "")
            if v not in ("", "NA", "nan", "None"):
                out[(r["chrom"], int(r["pos"]), r["ref"], r["alt"])] = float(v)
    return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def format_num(v: float | None, dec: int = 3) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "not reached"
    return f"{v:.{dec}f}"


def write_outputs(out_dir: Path, cohorts_dir: Path, raw_dir: Path, scores_dir: Path, results: dict[str, Any], runtime_s: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "calibration_summary.json"
    summary_path.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")

    def _tsv_val(v: float | None) -> Any:
        return "nan" if v is None else v

    tsv_path = out_dir / "calibration_tables.tsv"
    cols = [
        "stratum", "predictor", "role", "n_pos", "n_neg", "auroc", "auroc_lo", "auroc_hi",
        "auprc", "bp4_strong_4", "bp4_3", "bp4_mod_2", "bp4_supp_1",
        "pp3_supp_1", "pp3_mod_2", "pp3_3", "pp3_strong_4", "pp3_very_strong_8",
        "plp_reaching_pp3_supp_share", "plp_reaching_pp3_strong_share",
        "blb_reaching_bp4_supp_share", "blb_reaching_bp4_mod_share",
    ]
    with tsv_path.open("w", newline="") as h:
        w = csv.writer(h, delimiter="\t", lineterminator="\n")
        w.writerow(cols)
        for stratum, models in results["strata"].items():
            for key, m in models.items():
                disc_p = m["discounted_thresholds"]["PP3"]
                disc_b = m["discounted_thresholds"]["BP4"]
                p_reach = m["pathogenic_reach"]
                b_reach = m["benign_reach"]
                plp_supp = sum(p_reach[f"PP3_{n}"]["share"] for n, _ in POINTS_LEVELS)
                blb_supp = sum(b_reach[f"BP4_{n}"]["share"] for n, _ in POINTS_LEVELS)
                blb_mod = sum(b_reach[f"BP4_{n}"]["share"] for n, pts in POINTS_LEVELS if pts >= 2)
                plp_str = sum(p_reach[f"PP3_{n}"]["share"] for n, pts in POINTS_LEVELS if pts >= 4)
                w.writerow([
                    stratum, key, m["role"], m["n_pos"], m["n_neg"],
                    f"{m['auroc']:.4f}", f"{m['auroc_ci95'][0]:.4f}", f"{m['auroc_ci95'][1]:.4f}",
                    f"{m['auprc']:.4f}",
                    _tsv_val(disc_b["Strong_4"]), _tsv_val(disc_b["Intermediate_3"]),
                    _tsv_val(disc_b["Moderate_2"]), _tsv_val(disc_b["Supporting_1"]),
                    _tsv_val(disc_p["Supporting_1"]), _tsv_val(disc_p["Moderate_2"]),
                    _tsv_val(disc_p["Intermediate_3"]), _tsv_val(disc_p["Strong_4"]),
                    _tsv_val(disc_p["VeryStrong_8"]),
                    f"{plp_supp:.4f}", f"{plp_str:.4f}", f"{blb_supp:.4f}", f"{blb_mod:.4f}",
                ])

    lines = [
        "# ClinGen PP3/BP4 Score Calibration on Temporally Held-Out ClinVar Variants",
        "",
        "Calibrated following Pejaver et al. 2022 (`PMC9748256`, `alpha = 0.0441`, `c = 1124`) and",
        "Bergquist et al. 2025 (`PMC12208618`, adding +/-3pt intermediate levels and one-sided 95% conservative bootstrap bounds over `n_boot = 1000`).",
        "",
        "## Primary Stratum: ClinVar >=2-star missense SNVs held out from Dec 2023 (`post_2023_12`)",
        "",
        "| Predictor | Role | n P/LP | n B/LB | AUROC (95% CI) | BP4 Mod (-2) | BP4 Supp (-1) | PP3 Supp (+1) | PP3 Mod (+2) | PP3 Strong (+4) | P/LP >=+1 | B/LB <=-1 |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, m in results["strata"]["post_2023_12"].items():
        dec = next((p.decimals for p in PREDICTORS if p.key == key), 3)
        dp, db = m["discounted_thresholds"]["PP3"], m["discounted_thresholds"]["BP4"]
        plp_supp = sum(m["pathogenic_reach"][f"PP3_{n}"]["share"] for n, _ in POINTS_LEVELS)
        blb_supp = sum(m["benign_reach"][f"BP4_{n}"]["share"] for n, _ in POINTS_LEVELS)
        lines.append(
            f"| {m['label']} | {m['role']} | {m['n_pos']} | {m['n_neg']} | "
            f"{m['auroc']:.3f} ({m['auroc_ci95'][0]:.3f}, {m['auroc_ci95'][1]:.3f}) | "
            f"{format_num(db['Moderate_2'], dec)} | {format_num(db['Supporting_1'], dec)} | "
            f"{format_num(dp['Supporting_1'], dec)} | {format_num(dp['Moderate_2'], dec)} | "
            f"{format_num(dp['Strong_4'], dec)} | {100 * plp_supp:.1f}% | {100 * blb_supp:.1f}% |"
        )
    lines.extend([
        "",
        "## Allele-Frequency Controlled Stratum (`post_2023_12_af001`, gnomAD v3.1.2 grpmax < 0.1%)",
        "",
        "| Predictor | Role | n P/LP | n B/LB | AUROC (95% CI) | BP4 Mod (-2) | BP4 Supp (-1) | PP3 Supp (+1) | PP3 Mod (+2) | PP3 Strong (+4) |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ])
    for key, m in results["strata"]["post_2023_12_af001"].items():
        dec = next((p.decimals for p in PREDICTORS if p.key == key), 3)
        dp, db = m["discounted_thresholds"]["PP3"], m["discounted_thresholds"]["BP4"]
        lines.append(
            f"| {m['label']} | {m['role']} | {m['n_pos']} | {m['n_neg']} | "
            f"{m['auroc']:.3f} ({m['auroc_ci95'][0]:.3f}, {m['auroc_ci95'][1]:.3f}) | "
            f"{format_num(db['Moderate_2'], dec)} | {format_num(db['Supporting_1'], dec)} | "
            f"{format_num(dp['Supporting_1'], dec)} | {format_num(dp['Moderate_2'], dec)} | "
            f"{format_num(dp['Strong_4'], dec)} |"
        )
    lines.extend([
        "",
        "## Sensitivity Stratum (`post_2025_03`, held out from 2025-03-30 ClinVar)",
        "",
        "| Predictor | Role | n P/LP | n B/LB | AUROC | PP3 Supp (+1) | PP3 Strong (+4) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for key, m in results["strata"]["post_2025_03"].items():
        dec = next((p.decimals for p in PREDICTORS if p.key == key), 3)
        dp = m["discounted_thresholds"]["PP3"]
        lines.append(
            f"| {m['label']} | {m['role']} | {m['n_pos']} | {m['n_neg']} | "
            f"{m['auroc']:.3f} | {format_num(dp['Supporting_1'], dec)} | {format_num(dp['Strong_4'], dec)} |"
        )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")

    inputs = []
    manifest_path = raw_dir / "download_manifest.json"
    recorded_raw: dict[str, dict[str, Any]] = {}
    if manifest_path.exists():
        for entry in json.loads(manifest_path.read_text()):
            fname = entry.get("filename")
            if fname:
                recorded_raw[fname] = entry
    raw_seen: set[str] = set()
    for folder in (raw_dir, cohorts_dir, scores_dir):
        if not folder.exists():
            continue
        for p in sorted(folder.iterdir()):
            if p.is_file():
                inputs.append({"path": str(p.relative_to(out_dir.parent)), "bytes": p.stat().st_size, "sha256": sha256_file(p)})
                if folder == raw_dir:
                    raw_seen.add(p.name)
    for fname, entry in sorted(recorded_raw.items()):
        if fname not in raw_seen and "bytes" in entry and "sha256" in entry:
            inputs.append({"path": f"raw/{fname}", "bytes": entry["bytes"], "sha256": entry["sha256"]})
    inputs.sort(key=lambda item: item["path"])

    outputs = []
    for p in sorted(out_dir.iterdir()):
        if p.is_file() and p.name not in ("receipt.json", "SHA256SUMS"):
            outputs.append({"path": f"results/{p.name}", "bytes": p.stat().st_size, "sha256": sha256_file(p)})

    receipt = {
        "schema_version": "1.0",
        "kind": "clingen_pp3_bp4_cpu_calibration",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": round(runtime_s, 2),
        "seed": SEED,
        "parameters": {"alpha": ALPHA, "c_const": C_CONST, "minpoints": MINPOINTS, "gft": GFT, "n_boot": results["n_boot"]},
        "cost": {"paid_compute": False, "reason": "owned CPU workstation + precomputed public tables"},
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "inputs": inputs,
        "outputs": outputs,
    }
    receipt_path = out_dir / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    sums_lines = [f"{sha256_file(p)}  {p.name}\n" for p in sorted(out_dir.iterdir()) if p.is_file() and p.name != "SHA256SUMS"]
    (out_dir / "SHA256SUMS").write_text("".join(sums_lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, default=BASE_DIR)
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()

    t0 = time.time()
    cohorts_dir = args.workdir / "cohorts"
    scores_dir = args.workdir / "scores"
    raw_dir = args.workdir / "raw"
    out_dir = args.workdir / "results"

    c23 = load_cohort(cohorts_dir / "variants_post_2023_12.tsv")
    c25 = load_cohort(cohorts_dir / "variants_post_2025_03.tsv")
    g_ref = load_cohort(cohorts_dir / "gnomad_ref_missense.tsv")
    c23_af001 = [r for r in c23 if r.get("rare_af001") == "1"]

    strata = {
        "post_2023_12": c23,
        "post_2023_12_af001": c23_af001,
        "post_2025_03": c25,
    }

    results: dict[str, Any] = {
        "n_boot": args.n_boot,
        "seed": SEED,
        "parameters": {"alpha": ALPHA, "c_const": C_CONST, "minpoints": MINPOINTS, "gft": GFT},
        "posterior_cutoffs": posterior_cutoffs()[0],
        "benign_posterior_cutoffs": posterior_cutoffs()[1],
        "strata": {},
        "bergquist_2025_post_hoc": {},
    }

    g_keys = [(r["chrom"], int(r["pos"]), r["ref"], r["alt"]) for r in g_ref]
    for sname, rows in strata.items():
        results["strata"][sname] = {}
        results["bergquist_2025_post_hoc"][sname] = {}
        for spec in PREDICTORS:
            smap = load_score_map(scores_dir / spec.score_file, spec.column)
            if not smap:
                continue
            pos = np.array([smap[(r["chrom"], int(r["pos"]), r["ref"], r["alt"])] for r in rows if r["label"] == "1" and (r["chrom"], int(r["pos"]), r["ref"], r["alt"]) in smap], dtype=np.float64)
            neg = np.array([smap[(r["chrom"], int(r["pos"]), r["ref"], r["alt"])] for r in rows if r["label"] == "0" and (r["chrom"], int(r["pos"]), r["ref"], r["alt"]) in smap], dtype=np.float64)
            gs = np.array([smap[k] for k in g_keys if k in smap], dtype=np.float64)
            if len(pos) < 50 or len(neg) < 50:
                continue
            cal = calibrate_predictor(pos, neg, gs, spec.increment, spec.decimals, n_boot=args.n_boot, seed=SEED)
            cal["label"] = spec.label
            cal["role"] = spec.role
            cal["note"] = spec.note
            cal["coverage_pos"] = float(len(pos) / max(1, sum(r["label"] == "1" for r in rows)))
            cal["coverage_neg"] = float(len(neg) / max(1, sum(r["label"] == "0" for r in rows)))
            results["strata"][sname][spec.key] = cal
            if spec.key in BERGQUIST_2025_TABLE1:
                results["bergquist_2025_post_hoc"][sname][spec.key] = evaluate_bergquist_table1(spec.key, pos, neg)
            print(f"[{sname}] {spec.key}: n={len(pos)}+/{len(neg)}-, AUROC={cal['auroc']:.4f}, PP3+1={cal['discounted_thresholds']['PP3']['Supporting_1']}, BP4-1={cal['discounted_thresholds']['BP4']['Supporting_1']}", flush=True)

    write_outputs(out_dir, cohorts_dir, raw_dir, scores_dir, results, time.time() - t0)


if __name__ == "__main__":
    main()
