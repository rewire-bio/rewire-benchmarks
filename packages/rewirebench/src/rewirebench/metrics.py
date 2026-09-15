"""Scoring shared across rewire.it benchmarks.

For rare-positive prioritisation tasks the primary metric is precision at a fixed
review capacity, because a laboratory follows up a bounded number of candidates and
AUROC on a rare-positive set flatters everything. AUROC and average
precision are reported alongside, each named as computed: scikit-learn's
average_precision_score is not interpolated and is not the same quantity as a
trapezoidally integrated precision-recall area.

Uncertainty is a paired bootstrap over whole groups. Single-class resamples are
refused rather than filtered, because discarding them conditions the interval on
the draws that happened to contain both classes and reports it as narrower than
the data supports.
"""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def precision_at_n(labels, scores, n, rng=None):
    """Precision among the n highest-scoring variants.

    Ties at the cutoff are broken by a label-independent random order, so a method
    that emits many identical scores cannot gain from tie placement.
    """
    n = min(n, len(scores))
    rng = rng or np.random.default_rng(0)
    jitter = rng.permutation(len(scores))
    order = np.lexsort((jitter, -np.asarray(scores, dtype=float)))
    top = order[:n]
    return float(np.asarray(labels)[top].sum() / n)


def recall_at_n(labels, scores, n, rng=None):
    labels = np.asarray(labels)
    total = labels.sum()
    if total == 0:
        return 0.0
    n = min(n, len(scores))
    rng = rng or np.random.default_rng(0)
    jitter = rng.permutation(len(scores))
    order = np.lexsort((jitter, -np.asarray(scores, dtype=float)))
    return float(labels[order[:n]].sum() / total)


def point_metrics(labels, scores, capacity):
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    return {
        "n": int(len(labels)),
        "positives": int(labels.sum()),
        "prevalence": float(labels.mean()),
        "capacity": int(capacity),
        "precision_at_capacity": precision_at_n(labels, scores, capacity),
        "recall_at_capacity": recall_at_n(labels, scores, capacity),
        "average_precision_sklearn": float(average_precision_score(labels, scores)),
        "auroc": float(roc_auc_score(labels, scores)),
    }


def paired_group_bootstrap(labels, baseline, candidate, groups, capacity,
                           metric="precision_at_capacity", n=2000, seed=0,
                           max_single_class_frac=0.05):
    """95% interval for candidate minus baseline, resampling whole groups.

    Both methods score the same resampled rows, so the difference is paired.
    Raises if too many draws are single-class: with few independent groups the
    retained draws stop being a sample of anything.
    """
    labels = np.asarray(labels)
    baseline = np.asarray(baseline, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    groups = np.asarray(groups)

    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    index = {g: np.flatnonzero(groups == g) for g in uniq}

    def score(y, s, cap):
        if metric == "precision_at_capacity":
            return precision_at_n(y, s, cap)
        if metric == "average_precision_sklearn":
            return float(average_precision_score(y, s))
        if metric == "auroc":
            return float(roc_auc_score(y, s))
        raise ValueError(f"unknown metric {metric}")

    # The observed paired difference on the full common subset. This is the point
    # estimate. The resample mean below is theta-hat plus bootstrap bias, so
    # publishing it as "the effect" adds the bias rather than removing it.
    observed = score(labels, candidate, capacity) - score(labels, baseline, capacity)

    deltas, skipped, caps = [], 0, []
    for _ in range(n):
        drawn = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([index[g] for g in drawn])
        y = labels[rows]
        if y.min() == y.max():
            skipped += 1
            continue
        # Hold capacity proportional to the resampled size so the operating point
        # is comparable across draws of differing length.
        # Resampled lists vary in length because groups are unequal, so capacity is
        # held at a constant FRACTION of the list rather than a constant count. The
        # realised spread is reported so an interval is never read as "exactly N".
        cap = max(1, int(round(capacity * len(rows) / len(labels))))
        caps.append(cap)
        deltas.append(score(y, candidate[rows], cap) - score(y, baseline[rows], cap))

    if skipped > max_single_class_frac * n:
        raise ValueError(
            f"{skipped}/{n} resamples were single-class. Too few independent groups "
            f"({len(uniq)}) for this interval to mean anything; report the group count instead."
        )

    deltas = np.asarray(deltas)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    caps = np.asarray(caps)
    return {
        "metric": metric,
        "observed_delta": float(observed),
        "resample_mean_delta": float(deltas.mean()),
        "bootstrap_bias": float(deltas.mean() - observed),
        "mean_delta": float(deltas.mean()),  # deprecated alias, use observed_delta
        "realised_capacity": {
            "requested": int(capacity),
            "mean": float(caps.mean()),
            "min": int(caps.min()),
            "max": int(caps.max()),
            "list_fraction": round(float(capacity) / len(labels), 5),
        },
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "draws": int(n),
        "single_class_skipped": int(skipped),
        "independent_groups": int(len(uniq)),
    }
