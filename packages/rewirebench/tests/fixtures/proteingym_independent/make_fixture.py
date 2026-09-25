"""Generate the synthetic plmc_v2 models used by the EVCouplings parity fixture.

These represent no biological family. Parameters are hand-chosen or seeded, and
are attached to two real pinned-reference assays only so that the unmodified
ProteinGym scorer can read their coordinates. Writing follows the plmc_v2 layout
of EVCouplings CouplingsModel.to_file (e1362407). Run from this directory:

    python make_fixture.py
"""
import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
REFERENCE = HERE.parents[2] / "src/rewirebench/resources/proteingym/DMS_substitutions.csv"


def write_plmc_v2(path, *, alphabet, target, index_list, f_i, h_i, seed,
                  n_eff=150.25, lambda_h=0.01, theta=0.2):
    L, q = f_i.shape
    rng = np.random.default_rng(seed)
    weights = rng.uniform(0.2, 1.0, 7)
    pairs = L * (L - 1) // 2
    f_ij = rng.dirichlet(np.ones(q * q), pairs)
    J_ij = rng.normal(0, 0.5, (pairs, q * q))
    with open(path, "wb") as stream:
        np.array([L, q, 6, 1, 42], dtype="int32").tofile(stream)
        np.array([theta, lambda_h, 0.5, 0.0, n_eff], dtype="float32").tofile(stream)
        np.frombuffer(alphabet.encode(), "S1").tofile(stream)
        weights.astype("float32").tofile(stream)
        np.frombuffer(target.encode(), "S1").tofile(stream)
        np.array(index_list, dtype="int32").tofile(stream)
        f_i.astype("float32").tofile(stream)
        h_i.astype("float32").tofile(stream)
        f_ij.astype("float32").tofile(stream)
        J_ij.astype("float32").tofile(stream)


def site(alphabet, weights):
    """Frequencies from explicit symbol weights, with the remaining mass spread."""
    row = np.zeros(len(alphabet))
    for symbol, value in weights.items():
        row[alphabet.index(symbol)] = value
    rest = [i for i, s in enumerate(alphabet) if s not in weights]
    row[rest] = (1 - sum(weights.values())) / len(rest)
    return row


def main():
    with REFERENCE.open(newline="") as stream:
        reference = {r["DMS_id"]: r for r in csv.DictReader(stream)}
    fixtures = {}

    # KCNH2: MSA_start 535, so model index m is assay position m + 534.
    assay = "KCNH2_HUMAN_Kozek_2020"
    wt = reference[assay]["target_seq"]
    alphabet = "-ACDEFGHIKLMNPQRSTVY"  # plmc gap symbol; W deliberately absent
    index_list = [1, 3, 4]  # hole at model index 2 (assay position 536)
    positions = [m + 534 for m in index_list]
    target = "".join(wt[p - 1] for p in positions)
    a, b, c = target
    f_i = np.array([
        site(alphabet, {a: 0.45, "I": 0.45, "-": 0.02}),  # symmetric pair: equal fields
        site(alphabet, {b: 0.7, "K": 0.2, "E": 0.0, "P": 0.0, "-": 0.01}),  # wild type favoured
        site(alphabet, {c: 0.05, "F": 0.6, "-": 0.05}),  # alternative favoured
    ])
    rng = np.random.default_rng(13)
    write_plmc_v2(HERE / "kcnh2-gapped.model", alphabet=alphabet, target=target,
                  index_list=index_list, f_i=f_i, h_i=rng.normal(0, 1, f_i.shape), seed=1)
    p1, p2, p3 = positions
    fixtures[assay] = {
        "model": "kcnh2-gapped.model", "model_id": "KCNH2_HUMAN", "msa_start": 535,
        "scored": [f"{a}{p1}I", f"{b}{p2}K", f"{b}{p2}E", f"{c}{p3}F", f"{c}{p3}A",
                   f"{b}{p2}K:{c}{p3}F", f"{a}{p1}I:{b}{p2}E:{c}{p3}F"],
        "upstream_errors": {f"{wt[535]}536G": "missing middle model coordinate",
                            f"{c}{p3}W": "residue outside model alphabet",
                            f"{wt[533]}534A": "position before MSA_start",
                            f"{'A' if a != 'A' else 'C'}{p1}I": "wrong wild-type residue"},
    }

    # AMFR: MSA_start 1, 20 amino acids and no gap, two non-contiguous positions.
    assay = "AMFR_HUMAN_Tsuboyama_2023_4G3O"
    wt = reference[assay]["target_seq"]
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    index_list = [2, 5]
    target = "".join(wt[p - 1] for p in index_list)
    a, b = target
    f_i = np.array([site(alphabet, {a: 0.5, "W": 0.3}), site(alphabet, {b: 0.1, "G": 0.5})])
    write_plmc_v2(HERE / "amfr-20aa.model", alphabet=alphabet, target=target,
                  index_list=index_list, f_i=f_i, h_i=rng.normal(0, 1, f_i.shape), seed=2,
                  n_eff=37.5, lambda_h=0.04)
    fixtures[assay] = {
        "model": "amfr-20aa.model", "model_id": "AMFR_HUMAN", "msa_start": 1,
        "scored": [f"{a}2W", f"{a}2Y", f"{b}5G", f"{b}5W", f"{a}2W:{b}5G"],
        "upstream_errors": {f"{wt[2]}3A": "missing middle model coordinate"},
    }
    # ARGR: large N_eff, where SciPy's BFGS reports precision loss (warnflag 2)
    # at some sites; upstream accepts those fields silently.
    assay = "ARGR_ECOLI_Tsuboyama_2023_1AOY"
    wt = reference[assay]["target_seq"]
    alphabet = "-ACDEFGHIKLMNPQRSTVWY"
    index_list = [3, 4, 6, 9]
    target = "".join(wt[p - 1] for p in index_list)
    a, b, c, d = target
    alternatives = [next(s for s in order if s not in (x, "-")) for x, order in
                    zip(target, ("SAG", "YWF", "IVL", "PGA"))]
    s1, s2, s3, s4 = alternatives
    f_i = np.array([
        site(alphabet, {a: 0.6, s1: 0.3, "-": 0.02}),
        site(alphabet, {b: 0.5, s2: 0.3}),
        site(alphabet, {c: 0.9, s3: 0.05, **({"C": 0.0} if "C" not in (c, s3) else {})}),
        site(alphabet, {d: 0.25, s4: 0.25, "-": 0.1}),
    ])
    write_plmc_v2(HERE / "argr-precision-loss.model", alphabet=alphabet, target=target,
                  index_list=index_list, f_i=f_i, h_i=rng.normal(0, 1, f_i.shape), seed=3,
                  n_eff=4e5, lambda_h=0.01)
    fixtures[assay] = {
        "model": "argr-precision-loss.model", "model_id": "ARGR_ECOLI", "msa_start": 1,
        "scored": [f"{a}3{s1}", f"{b}4{s2}", f"{c}6{s3}", f"{d}9{s4}", f"{a}3{s1}:{d}9{s4}"],
        "upstream_errors": {f"{wt[4]}5{'A' if wt[4] != 'A' else 'G'}": "missing middle model coordinate"},
    }
    (HERE / "fixture.json").write_text(json.dumps(fixtures, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
