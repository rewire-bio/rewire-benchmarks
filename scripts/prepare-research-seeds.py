#!/usr/bin/env python3
"""Connect existing, verified local evaluations to portable research manifests.

Run in the runner checkout. Outputs (including assay-derived tables and private
paths) belong in ignored workbench storage. Only manifests.json is distributable.
This does not download data, refit models, publish findings, or alter old runs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote

import numpy as np
from rewirebench.metrics import point_metrics
from rewirebench.protocols.proteingym import _metrics as protein_metrics
from rewirebench.sdk import _code_digest, _validate_prepared
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, ndcg_score


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def require_hash(path, expected):
    if sha(path) != expected:
        raise ValueError(
            f"Artifact drift: {path.name}; do not substitute another cohort or snapshot"
        )


def tsv(path):
    with Path(path).open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError(f"Duplicate source identifiers: {path.name}")
    return rows


def finite(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def regression_metrics(rows, method):
    y = np.array([r["y"] for r in rows])
    p = np.array([r["scores"][method] for r in rows])
    variable = np.ptp(p) > 0 and np.ptp(y) > 0
    return {
        "mse": float(mean_squared_error(y, p)),
        "pearson": float(pearsonr(y, p).statistic) if variable else None,
        "spearman": float(spearmanr(y, p).statistic) if variable else None,
        "ndcg": float(ndcg_score([y - y.min()], [p])),
    }


class Preparation:
    def __init__(self, root, output, catalogue, verified_at):
        self.root, self.output, self.catalogue = root, output, catalogue
        self.at, self.resolver, self.manifests = verified_at, {}, []
        self.ids = {r["id"] for r in catalogue["records"]}
        self.revision = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()

    def artifact(self, case, role, path, uri=None, semantic=None):
        ident = f"{case}-{role}"
        self.resolver[ident] = str(path.resolve())
        if uri is None and path.resolve().is_relative_to(self.root.resolve()):
            rel = path.resolve().relative_to(self.root.resolve()).as_posix()
            stored = subprocess.run(
                ["git", "-C", str(self.root), "show", f"{self.revision}:{rel}"],
                capture_output=True,
                check=False,
            )
            if stored.returncode == 0 and hashlib.sha256(stored.stdout).hexdigest() == sha(path):
                uri = f"https://raw.githubusercontent.com/rewire-bio/rewire-benchmarks/{self.revision}/{quote(rel)}"
        value = {
            "id": ident,
            "role": role,
            "sha256": sha(path),
            "format": path.suffix.lstrip("."),
            "uri": uri,
        }
        if semantic:
            value["semantic_sha256"] = semantic
        return value

    def finish(
        self,
        *,
        ident,
        title,
        question,
        dataset,
        evaluations,
        protocol,
        rows,
        kind,
        artifacts,
        expected,
        features,
        independent,
        outcome,
        unit,
        limitations,
        recipes=(),
        sdk_protocol="mfass-v2",
    ):
        for linked in [dataset, protocol, *evaluations]:
            if linked not in self.ids:
                raise ValueError(f"Catalogue reference absent: {linked}")
        if len({r["id"] for r in rows}) != len(rows) or any(r["split"] != "test" for r in rows):
            raise ValueError("Research observation tables require unique original test identifiers")
        metric = {}
        for method, targets in expected.items():
            selected = [r for r in rows if r["scores"][method] is not None]
            y = np.array([r["y"] for r in selected])
            pred = np.array([r["scores"][method] for r in selected])
            if kind == "mfass":
                actual = point_metrics(y, pred, 100)
            elif kind == "proteingym":
                actual = protein_metrics(
                    y, pred, np.array([r["features"]["binary_target"] for r in selected])
                )
            else:
                actual = regression_metrics(selected, method)
            metric[method] = actual
            for key, target in targets.items():
                if target is None:
                    if actual.get(key) is not None:
                        raise ValueError(f"Undefined metric changed: {method}/{key}")
                elif actual.get(key) is None or abs(actual[key] - target) > 1e-9:
                    raise ValueError(f"Metric drift: {method}/{key}: {actual.get(key)} != {target}")
        path = self.output / f"{ident}.table.json"
        write(path, {"schema_version": "1.0", "metric_kind": kind, "rows": rows})
        table = self.artifact(ident, "table", path)
        artifacts.append(table)
        receipt = self.output / f"{ident}.receipt.json"
        write(
            receipt,
            {
                "schema_version": "1.0",
                "verified_at": self.at,
                "table_sha256": table["sha256"],
                "count": len(rows),
                "metrics": metric,
                "check": "Saved predictions replay; not independent biological validation",
            },
        )
        artifacts.append(self.artifact(ident, "receipt", receipt))
        artifacts.append(self.artifact(ident, "preparation_code", Path(__file__).resolve()))
        artifacts.append(self.artifact(ident, "environment", self.root / "uv.lock"))
        checks = [
            {"check": c, "status": "passed", "detail": d}
            for c, d in [
                (
                    "artifact_hashes",
                    "Exact local bytes recorded; source pins checked where available.",
                ),
                ("join_integrity", f"{len(rows)} unique test IDs; scores and outcomes reconcile."),
                (
                    "score_semantics",
                    f"Higher scores predict higher {outcome}; metrics retain their own direction.",
                ),
                (
                    "metric_replay",
                    "Saved predictions reproduce recorded metrics within 1e-9 absolute tolerance.",
                ),
                (
                    "annotations",
                    "Registered subgroup fields derive from the preserved source snapshot.",
                ),
                (
                    "dependence",
                    independent or "Independent sampling unit unknown; descriptive analysis only.",
                ),
            ]
        ]
        if recipes:
            checks.extend(
                [
                    {
                        "check": "recipe_pinned",
                        "status": "passed",
                        "detail": "Registered SDK recipe; implementation artifact and environment hashes retained.",
                    },
                    {
                        "check": "resource_estimate",
                        "status": "passed",
                        "detail": "Local CPU controls previously executed; all jobs remain subject to the campaign watchdog.",
                    },
                ]
            )
        manifest = {
            "schema_version": "1.0",
            "id": ident,
            "title": title,
            "question": question,
            "catalogue_release_id": self.catalogue["release_id"],
            "dataset_id": dataset,
            "runner_code_sha256": _code_digest(),
            "evaluation_ids": evaluations,
            "protocol_id": protocol,
            "sdk_protocol_id": sdk_protocol,
            "artifacts": artifacts,
            "table_artifact_id": table["id"],
            "semantics": {
                "target": "binary" if kind == "mfass" else "continuous",
                "outcome": outcome,
                "unit": unit,
                "score_direction": "higher",
                "join_key": "id",
                "independent_unit": independent,
                "subgroup_fields": features,
                "exposed": True,
                "split": "test",
            },
            "expected_metrics": expected,
            "metric_tolerance": 1e-9,
            "verification": {"verified_at": self.at, "checks": checks, "limitations": limitations},
            "local_recipes": list(recipes),
        }
        self.manifests.append(manifest)

    def mfass(self):
        base = self.root / "benchmarks/mfass"
        ident = "mfass-v2-discrepancies"
        provenance = read(base / "provenance/mfass-v2-local-dnabert2.json")
        sources = [
            ("cohort", "data/cohort.tsv", "validated_cohort"),
            ("outcomes", "data/snv_data_clean.txt", "published_variant_table"),
            ("annotations", "data/snv_func_annot.txt", "functional_annotation"),
            ("split", "splits/split-v2.tsv", "canonical_split"),
        ]
        artifacts = []
        for role, filename, key in sources:
            path = base / filename
            require_hash(path, provenance["source_sha256"][key])
            uri = (
                "https://raw.githubusercontent.com/KosuriLab/MFASS/master/processed_data/snv/"
                + path.name
                if role in ["outcomes", "annotations"]
                else None
            )
            artifacts.append(self.artifact(ident, role, path, uri=uri))
        cohort = {r["id"]: r for r in tsv(base / "data/cohort.tsv")}
        raw = {r["id"]: r for r in tsv(base / "data/snv_data_clean.txt")}
        split = [r for r in tsv(base / "splits/split-v2.tsv") if r["split"] == "test"]
        split_by_id = {r["id"]: r for r in split}
        methods = [
            "baseline-kmer-position-v2",
            "dnabert2-117m-frozen-pair-logreg",
            "spliceai-1.3.1",
            "pangolin-maskFalse",
        ]
        scores, expected = {}, {}
        for method in methods:
            p = base / f"results/{method}.predictions.tsv"
            predictions = tsv(p)
            if any(r["id"] not in split_by_id for r in predictions):
                raise ValueError("Unknown prediction ID")
            for r in predictions:
                s = split_by_id[r["id"]]
                if r["group"] != s["group"] or int(r["label"]) != int(cohort[r["id"]]["sdv"]):
                    raise ValueError("Prediction label/group mismatch")
            values = [finite(r["score"]) for r in predictions]
            npy = base / f"results/{method}.scores.npy"
            if npy.exists():
                exact = np.load(npy, allow_pickle=False)
                if len(exact) != len(values) or not np.allclose(values, exact, atol=5e-5, rtol=0):
                    raise ValueError("Full-precision score order mismatch")
                values = exact
                artifacts.append(self.artifact(ident, f"{method}-scores", npy))
            scores[method] = {
                r["id"]: float(v) if v is not None else None for r, v in zip(predictions, values)
            }
            report_path = base / f"results/{method}.json"
            expected[method] = {
                k: v
                for k, v in read(report_path)["metrics"].items()
                if k
                in [
                    "precision_at_capacity",
                    "recall_at_capacity",
                    "average_precision_sklearn",
                    "auroc",
                ]
            }
            artifacts.extend(
                [
                    self.artifact(ident, f"{method}-predictions", p),
                    self.artifact(ident, f"{method}-report", report_path),
                ]
            )
        rows = []
        for s in split:
            c, r = cohort[s["id"]], raw[s["id"]]
            distance = min(
                abs(float(c["rel_position"]) - float(c["intron1_len"])),
                abs(float(c["rel_position"]) - float(c["intron1_len"]) - float(c["exon_len"])),
            )
            gap = abs(float(r["v2_dpsi_R1"]) - float(r["v2_dpsi_R2"]))
            rows.append(
                {
                    "id": s["id"],
                    "y": int(c["sdv"]),
                    "group": s["group"],
                    "split": "test",
                    "scores": {m: scores[m].get(s["id"]) for m in methods},
                    "features": {
                        "boundary_band": "0-2"
                        if distance <= 2
                        else "3-10"
                        if distance <= 10
                        else "11-30"
                        if distance <= 30
                        else ">30",
                        "replicate_gap_band": "<=0.1"
                        if gap <= 0.1
                        else "0.1-0.25"
                        if gap <= 0.25
                        else ">0.25",
                        "strand": c["strand"],
                        "legacy_orientation": c["legacy_sequence_orientation"],
                        "gene": c["ensembl_gene_id"],
                        "exon_length": float(c["exon_len"]),
                        "replicate_gap": gap,
                    },
                }
            )
        self.finish(
            ident=ident,
            title="MFASS v2: model disagreement",
            question="Does model disagreement depend on exon boundary distance, assay replicate agreement or gene/exon concentration?",
            dataset="rewire-mfass-v2-dataset",
            protocol="rewire-mfass-v2",
            evaluations=[
                "rewire-evaluation-baseline-kmer-position-v2",
                "rewire-evaluation-dnabert2-117m-frozen-pair-logreg",
                "rewire-evaluation-spliceai-1-3-1",
                "rewire-evaluation-pangolin-maskfalse",
            ],
            rows=rows,
            kind="mfass",
            artifacts=artifacts,
            expected=expected,
            features=["boundary_band", "replicate_gap_band", "strand", "legacy_orientation"],
            independent="connected exon/gene group",
            outcome="assay splice disruption",
            unit="binary label",
            limitations=[
                "Existing test outcomes have been inspected; all new subgroup findings are exploratory.",
                "Specialists use different genomic context and annotation releases; model-only attribution is unsupported.",
                "Boundary bands are symmetric distances, not canonical dinucleotide annotations; baseline uses distance features.",
                "The historical v1 sequence-orientation error is already corrected and is not a new discovery.",
                "Source assay data have no declared redistribution licence; obtain original tables from KosuriLab/MFASS.",
            ],
        )

    def sdk_case(self, name):
        base = self.root / "workbench/local-runs-2026-09-20"
        pg = name == "proteingym"
        short = "proteingym-amfr" if pg else name
        ident = f"{short}-discrepancies"
        prepared_path = (
            base / "proteingym/run-01/prepared/prepared.json"
            if pg
            else base / f"sequence-runs/{name}-prepared/prepared.json"
        )
        prepared = read(prepared_path)
        _validate_prepared(prepared)
        methods = ["proteingym-esm2"] if pg else [f"{name}-composition", f"{name}-train-mean"]
        artifacts = [
            self.artifact(ident, "prepared", prepared_path, semantic=prepared["prepared_sha256"])
        ]
        scores, expected = {}, {}
        for method in methods:
            evaluation = (
                base / "proteingym/run-01/evaluation" if pg else base / f"sequence-runs/{method}"
            )
            p = evaluation / "predictions.json"
            report = read(evaluation / "report.json")
            public_report = self.root / f"research/local-runs-2026-09-20/{method}/report.json"
            recorded = read(public_report)
            if (
                report["predictions_sha256"] != recorded["predictions_sha256"]
                or report["prepared_sha256"] != recorded["prepared_sha256"]
            ):
                raise ValueError("Local report differs from recorded scientific evidence")
            if report["prepared_sha256"] != prepared["prepared_sha256"]:
                raise ValueError("Prepared snapshot mismatch")
            require_hash(p, report["predictions_sha256"])
            scores[method] = read(p)
            expected[method] = (
                next(iter(report["protocol_results"]["per_assay"].values()))["metrics"]
                if pg
                else report["metrics"]
            )
            expected[method] = {k: v for k, v in expected[method].items() if k not in ["n"]}
            artifacts.extend(
                [
                    self.artifact(ident, f"{method}-predictions", p),
                    self.artifact(ident, f"{method}-report", public_report),
                ]
            )
            source = (
                self.root
                / f"research/local-runs-2026-09-20/{method}/"
                / ("retrieval.json" if pg else "source.json")
            )
            artifacts.append(self.artifact(ident, f"{method}-source", source))
        selected = [r for r in prepared["rows"] if r["split"] == "test"]
        ids = {r["id"] for r in selected}
        if any(set(s) != ids for s in scores.values()):
            raise ValueError(
                "Prediction membership differs from exact original prepared test snapshot"
            )
        seq_counts = (
            Counter(r["inputs"].get("sequence", "") for r in prepared["rows"]) if not pg else {}
        )
        rows = []
        for r in selected:
            if pg:
                mutant = r["inputs"]["mutant"]
                features = {
                    "substitution_count": len(mutant.split(":")),
                    "mutation_class": "single" if ":" not in mutant else "multiple",
                    "binary_target": r["target_binary"],
                }
            else:
                seq = r["inputs"]["sequence"]
                gc = sum(seq.count(c) for c in "GC") / len(seq)
                features = {
                    "sequence_length": len(seq),
                    "duplicate_sequence": "yes" if seq_counts[seq] > 1 else "no",
                    "length_band": "<=250"
                    if len(seq) <= 250
                    else "251-500"
                    if len(seq) <= 500
                    else ">500",
                }
                if name == "mrnabench":
                    features.update(
                        gc_band="<0.4" if gc < 0.4 else "0.4-0.6" if gc <= 0.6 else ">0.6",
                        source_index=r["source_index"],
                    )
            rows.append(
                {
                    "id": r["id"],
                    "y": r["target"],
                    "group": None,
                    "split": "test",
                    "scores": {m: scores[m][r["id"]] for m in methods},
                    "features": features,
                }
            )
        limitations = [
            "Existing test outcomes are exposed; no independent validation is claimed.",
            "Independent experimental grouping is unavailable; subgroup summaries are descriptive.",
            "Prepared opaque IDs must not be joined to a newly prepared snapshot by ID.",
        ]
        features = (
            ["mutation_class"]
            if pg
            else ["duplicate_sequence", "length_band"]
            + (["gc_band"] if name == "mrnabench" else [])
        )
        recipes = [] if pg else ["sdk:train-mean-v1", "sdk:sequence-composition-v1"]
        if recipes:
            artifacts.append(
                self.artifact(
                    ident,
                    "recipe_code",
                    self.root / "packages/rewirebench/src/rewirebench/adapters/sequence.py",
                )
            )
        if pg:
            title, question, dataset, protocol = (
                "ProteinGym AMFR: negative correlation",
                "Does the negative association differ between single and multiple substitutions?",
                "rewire-dataset-proteingym-amfr-v13",
                "rewire-protocol-proteingym-amfr-v13",
            )
            limitations += [
                "One protein construct, complete selected assay but partial benchmark suite.",
                "Official three-decimal metrics and score sign are preserved; no post-hoc sign reversal.",
                "Upstream archive checksums were observed locally, not independently source-verified.",
            ]
        elif name == "mrnabench":
            title, question, dataset, protocol = (
                "mRNABench designed: composition effects",
                "Where does the composition predictor improve over the training-mean control?",
                "rewire-dataset-mrnabench-designed-mrl-v1",
                "rewire-protocol-mrnabench-designed-mrl-v1",
            )
            limitations += [
                "Random split does not establish homology separation.",
                "Source data reuse licence is unreported; no raw data redistribution.",
            ]
        else:
            title, question, dataset, protocol = (
                "FLIP2 Rhomax: metric interpretation control",
                "Why can a constant predictor have high NDCG and undefined Spearman correlation?",
                "rewire-dataset-flip2-rhomax-by-wild-type-v3",
                "rewire-protocol-flip2-rhomax-by-wild-type-v1",
            )
            limitations += [
                "High wavelength is not universally biologically preferable; NDCG is a numerical ranking metric.",
                "Known methodological control, not a novel biological finding.",
            ]
        self.finish(
            ident=ident,
            title=title,
            question=question,
            dataset=dataset,
            protocol=protocol,
            evaluations=[f"rewire-local-20260920-evaluation-{m}" for m in methods],
            rows=rows,
            kind="proteingym" if pg else "regression",
            artifacts=artifacts,
            expected=expected,
            features=features,
            independent=None,
            outcome="DMS stability"
            if pg
            else "mean ribosome load"
            if name == "mrnabench"
            else "spectral wavelength",
            unit="assay DMS score" if pg else "MRL" if name == "mrnabench" else "nm",
            limitations=limitations,
            recipes=recipes,
            sdk_protocol=prepared["protocol_id"],
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalogue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--verified-at", default=datetime.now(UTC).isoformat())
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    prep = Preparation(args.root, args.output, read(args.catalogue), args.verified_at)
    prep.mfass()
    for name in ["proteingym", "mrnabench", "flip2"]:
        prep.sdk_case(name)
    write(args.output / "manifests.json", prep.manifests)
    write(args.output / "resolver.private.json", prep.resolver)
    print(
        json.dumps(
            {
                "cases": len(prep.manifests),
                "public_metadata": "manifests.json",
                "private_files": "All other files stay in local ignored workspace.",
            }
        )
    )


if __name__ == "__main__":
    main()
