"""Grouping and splitting.

The grouping unit is the connected component of the relations that must not cross
the train/test boundary, never a concatenated key. Concatenating keys reports more
independent units than exist whenever one value of one key spans two values of
another, and the error is invisible in the resulting file.
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Any, Hashable, Iterable, Mapping, Sequence


def connected_components(
    records: Sequence[Mapping[str, Any]],
    keys: Sequence[str],
    id_field: str = "id",
) -> dict[str, Hashable]:
    """Map record id to its component root over the given equivalence keys."""
    parent: dict[Hashable, Hashable] = {}

    def find(x: Hashable) -> Hashable:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: Hashable, b: Hashable) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for rec in records:
        node = ("rec", rec[id_field])
        find(node)
        for k in keys:
            union(node, (k, rec[k]))

    roots = {rec[id_field]: find(("rec", rec[id_field])) for rec in records}
    return _stabilise(roots)


def _stabilise(roots: dict[str, Hashable]) -> dict[str, str]:
    """Replace arbitrary component roots with stable, compact ids.

    The union-find root is whichever node happened to win, which is not stable
    across insertion order and serialises as a Python repr. Components are instead
    numbered in sorted order of their smallest member id, so the split file is
    compact, diffable, and identical across runs and platforms.
    """
    members: dict[Hashable, list[str]] = {}
    for rid, root in roots.items():
        members.setdefault(root, []).append(rid)
    order = sorted(members, key=lambda r: min(members[r]))
    width = max(4, len(str(len(order))))
    return {
        rid: f"g{i:0{width}d}"
        for i, root in enumerate(order, start=1)
        for rid in members[root]
    }


def naive_key_count(records: Iterable[Mapping[str, Any]], keys: Sequence[str]) -> int:
    """How many units a concatenated key would claim. Report beside the real count."""
    return len({tuple(r[k] for k in keys) for r in records})


def assign_groups(
    records: Sequence[Mapping[str, Any]],
    groups: Mapping[str, Hashable],
    label_field: str,
    test_frac: float = 0.3,
    seed: int = 0,
    restarts: int = 200,
    id_field: str = "id",
) -> tuple[dict[str, str], dict[Hashable, list], dict[str, Any]]:
    """Hold out whole components, matching prevalence between arms.

    Components are assigned entire, so achieved prevalence is whatever the drawn
    groups give. A single draw can land far from the cohort rate, so this takes the
    best of `restarts` seeded draws on closeness of test prevalence to cohort
    prevalence. That is stratification on the labels' marginal rate only, fixed
    before any model runs, and never on a prediction.
    """
    if not 0.0 < test_frac < 1.0:
        raise ValueError("test_frac must lie strictly between 0 and 1")

    by_group: dict[Hashable, list] = defaultdict(list)
    for r in records:
        by_group[groups[r[id_field]]].append(r)

    keys = sorted(by_group, key=str)
    n_target = len(records) * test_frac
    cohort_prev = sum(int(r[label_field]) for r in records) / len(records)

    best: tuple[float, set] | None = None
    for attempt in range(restarts):
        rng = random.Random(seed + attempt)
        order = keys[:]
        rng.shuffle(order)
        chosen: set = set()
        n = 0
        for g in order:
            if n >= n_target:
                break
            chosen.add(g)
            n += len(by_group[g])
        pos = sum(int(r[label_field]) for g in chosen for r in by_group[g])
        prev = pos / n if n else 0.0
        score = abs(prev - cohort_prev)
        if best is None or score < best[0]:
            best = (score, chosen)

    assert best is not None
    test_groups = best[1]
    split = {}
    for g, rs in by_group.items():
        arm = "test" if g in test_groups else "train"
        for r in rs:
            split[r[id_field]] = arm

    def arm_stats(which: str) -> dict[str, Any]:
        rs = [r for r in records if split[r[id_field]] == which]
        pos = sum(int(r[label_field]) for r in rs)
        return {
            "variants": len(rs),
            "positives": pos,
            "prevalence": round(pos / len(rs), 5) if rs else 0.0,
            "groups": len({groups[r[id_field]] for r in rs}),
        }

    sizes = Counter(len(v) for v in by_group.values())
    manifest = {
        "independent_groups": len(by_group),
        "largest_group": max(len(v) for v in by_group.values()),
        "median_group": sorted(len(v) for v in by_group.values())[len(by_group) // 2],
        "singleton_groups": sizes.get(1, 0),
        "seed": seed,
        "restarts": restarts,
        "test_frac_requested": test_frac,
        "cohort_prevalence": round(cohort_prev, 5),
        "train": arm_stats("train"),
        "test": arm_stats("test"),
    }
    return split, dict(by_group), manifest
