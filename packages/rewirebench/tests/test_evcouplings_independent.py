"""EVCouplings independent model: upstream parity, coordinates, coverage and leakage.

Parity values come from upstream-receipt.json, produced by running the unmodified
pinned ProteinGym score_mutants.py and EVCouplings to_independent_model() in a
separate environment (scripts/baseline_parity/evcouplings_upstream_receipt.py).
Setting REWIRE_EVCOUPLINGS_PYTHON and REWIRE_EVCOUPLINGS_UPSTREAM reruns it live.
"""
import copy
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from rewirebench import cli, sdk
from rewirebench.adapters import evcouplings_independent as evc
from rewirebench.baselines import run_baselines
from rewirebench.protocols import proteingym

FIXTURE = Path(__file__).parent / "fixtures" / "proteingym_independent"
RECEIPT = json.loads((FIXTURE / "upstream-receipt.json").read_text())
SPEC = json.loads((FIXTURE / "fixture.json").read_text())
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "baseline_parity"))
import evcouplings_compare as compare

R = "ARGR_ECOLI_Tsuboyama_2023_1AOY"  # ill-conditioned precision-loss stress fixture
STABLE = ("AMFR_HUMAN_Tsuboyama_2023_4G3O", "KCNH2_HUMAN_Kozek_2020")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wild_type(assay):
    with proteingym.resource_path("DMS_substitutions.csv").open(newline="") as stream:
        return next(r for r in csv.DictReader(stream) if r["DMS_id"] == assay)["target_seq"]


def manifest(tmp_path, **changes):
    provenance = {k: "synthetic fixture; not a biological model" for k in evc.PROVENANCE_FIELDS}
    provenance["alignment_sha256"] = "0" * 64
    assays = {}
    for assay, spec in SPEC.items():
        assays[assay] = {"model_path": str(FIXTURE / spec["model"]), "format": "plmc_v2",
                         "model_sha256": sha(FIXTURE / spec["model"]), "precision": "float32",
                         "model_id": spec["model_id"], "msa_start": spec["msa_start"],
                         "provenance": dict(provenance)}
    data = {"schema": evc.MANIFEST_SCHEMA, "dms_labels_used": False, "assays": assays}
    for assay, fields in changes.items():
        data["assays"][assay].update(fields)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(data))
    return path


@pytest.fixture
def artifact(tmp_path):
    return evc.prepare_artifact(manifest(tmp_path), tmp_path / "artifact.json")


@pytest.fixture
def adapter(artifact):
    return evc.EVCouplingsIndependent(artifact["artifact"], artifact["artifact_sha256"])


def rows(assay, mutants):
    wt = wild_type(assay)
    return [{"id": f"{assay}::{m}", "assay_id": assay, "wild_type_sequence": wt, "mutant": m}
            for m in mutants]


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6)


def test_receipt_describes_current_fixture_and_pins():
    assert RECEIPT["reference_sha256"] == proteingym.REFERENCE_SHA256
    assert RECEIPT["fixture_json_sha256"] == sha(FIXTURE / "fixture.json")
    assert RECEIPT["upstream_revisions"] == {"evcouplings": evc.EVCOUPLINGS_REVISION,
                                             "proteingym": proteingym.UPSTREAM_REVISION}
    pinned = {path.split("/", 1)[1]: digest for path, digest in RECEIPT["upstream_source_sha256"].items()}
    assert pinned == {k: v["sha256"] for k, v in evc.UPSTREAM_SOURCES.items()}
    for assay, spec in SPEC.items():
        entry = RECEIPT["assays"][assay]
        assert entry["model_sha256"] == sha(FIXTURE / spec["model"])
        assert entry["J_ij_cleared"] is True
        assert entry["stdout_offset"] == [f"Offset: {spec['msa_start'] - 1}"]


def adapter_report(artifact, adapter, receipt):
    """Compare the adapter with a receipt over all assays, sites and receipt mutants."""
    assays = json.loads(Path(artifact["artifact"]).read_text())["assays"]
    predictions = {}
    for assay, spec in SPEC.items():
        assert set(receipt["assays"][assay]["prediction_independent"]) == set(spec["scored"])
        scored = adapter.predict(rows(assay, spec["scored"]))
        predictions[assay] = {k.split("::", 1)[1]: v for k, v in scored.items()}
    return compare.compare(assays, receipt["assays"], SPEC, predictions)


def test_stable_fixtures_match_frozen_upstream_receipt(artifact, adapter):
    """AMFR and KCNH2: every raw field, WT contrast and score against the frozen receipt."""
    report, failures = adapter_report(artifact, adapter, RECEIPT)
    stable = [f for f in failures if f.startswith(STABLE)]
    assert not stable, json.dumps({"failures": stable, "report": {a: report[a] for a in STABLE}}, indent=1)


def test_stress_fixture_selected_scores_match_frozen_receipt(artifact, adapter):
    """ARGR (N_eff 4e5): the frozen selected scores must hold in every runtime at 1e-6.

    Its raw fields and full contrasts are checked against upstream executed in
    the same runtime (test_same_runtime_upstream_parity). Against the frozen
    macOS receipt they are reported on failure, not asserted, because this
    ill-conditioned fit's iterate depends on the numerical runtime.
    """
    report, failures = adapter_report(artifact, adapter, RECEIPT)
    scores = [f for f in failures if f.startswith(R) and ": score " in f]
    assert not scores, json.dumps({"failures": scores, "report": report[R]}, indent=1)


def test_direction_symmetry_and_additivity(adapter):
    kcnh2 = "KCNH2_HUMAN_Kozek_2020"
    p = adapter.predict(rows(kcnh2, SPEC[kcnh2]["scored"]))
    s = {k.split("::")[1]: v for k, v in p.items()}
    assert s["K538F"] > 0          # frequency 0.05 -> 0.6: favoured substitution
    assert s["R537K"] < 0 and s["R537E"] < s["R537K"]  # 0.7 -> 0.2 -> absent
    assert abs(s["V535I"]) < 1e-6  # equal frequencies: analytic zero
    assert s["R537K:K538F"] == pytest.approx(s["R537K"] + s["K538F"], abs=1e-12)
    assert s["V535I:R537E:K538F"] == pytest.approx(s["V535I"] + s["R537E"] + s["K538F"], abs=1e-12)


def test_not_the_epistatic_model_with_couplings_cleared(adapter):
    """Original plmc fields differ from the refit; using them would be a different method."""
    assay = "AMFR_HUMAN_Tsuboyama_2023_4G3O"
    model = evc.read_plmc_v2(FIXTURE / SPEC[assay]["model"])
    L, q = 2, 20  # h_i follows header, alphabet, 7 weights, target, index list and f_i
    offset = 20 + 20 + q + 7 * 4 + L + 4 * L + L * q * 4
    original_h = np.frombuffer((FIXTURE / SPEC[assay]["model"]).read_bytes(), "float32", L * q, offset).reshape(L, q)
    alphabet = model["alphabet"]
    cleared = float(original_h[1, alphabet.index("G")] - original_h[1, alphabet.index("Q")])
    ours = adapter.predict(rows(assay, ["Q5G"]))[f"{assay}::Q5G"]
    assert not close(cleared, ours)
    assert close(ours, RECEIPT["assays"][assay]["prediction_independent"]["Q5G"])


def test_upstream_failures_become_unscored_or_errors_never_zero(adapter):
    reasons = {"missing middle model coordinate": evc.UNSCORED_POSITION,
               "position before MSA_start": evc.UNSCORED_POSITION,
               "residue outside model alphabet": evc.UNSCORED_RESIDUE}
    for assay, spec in SPEC.items():
        for mutant, kind in spec["upstream_errors"].items():
            assert RECEIPT["assays"][assay]["invalid_mutants"][mutant]["type"] == "ValueError"
            if kind == "wrong wild-type residue":
                with pytest.raises(ValueError, match="does not match"):
                    adapter.predict(rows(assay, [mutant]))
                continue
            result = adapter.predict(rows(assay, [mutant]))[f"{assay}::{mutant}"]
            assert result == {"score": None, "reason": reasons[kind]}
    # One uncovered component makes the whole multiple mutant unscored.
    assay = "KCNH2_HUMAN_Kozek_2020"
    double = adapter.predict(rows(assay, ["K538F:A536G"]))[f"{assay}::K538F:A536G"]
    assert double["score"] is None
    with proteingym.resource_path("DMS_substitutions.csv").open(newline="") as stream:
        other = next(r for r in csv.DictReader(stream) if r["DMS_id"] not in SPEC)
    wt = other["target_seq"]
    row = {"id": "x", "assay_id": other["DMS_id"], "wild_type_sequence": wt,
           "mutant": f"{wt[0]}1{'A' if wt[0] != 'A' else 'C'}"}
    assert adapter.predict([row])["x"] == {"score": None, "reason": evc.UNSCORED_ASSAY}


def test_inputs_are_allowlisted_and_row_batch_order_invariant(adapter):
    assay = "KCNH2_HUMAN_Kozek_2020"
    batch = rows(assay, SPEC[assay]["scored"])
    whole = adapter.predict(batch)
    singles = {}
    for row in reversed(batch):
        singles.update(adapter.predict([row]))
    assert singles == whole
    leaked = copy.deepcopy(batch[:1])
    leaked[0]["target"] = 1.0
    with pytest.raises(ValueError, match="labels"):
        adapter.predict(leaked)
    wrong = copy.deepcopy(batch[:1])
    sequence = wrong[0]["wild_type_sequence"]
    wrong[0]["wild_type_sequence"] = sequence[:9] + ("A" if sequence[9] != "A" else "C") + sequence[10:]
    with pytest.raises(ValueError, match="wild type"):
        adapter.predict(wrong)
    assert not hasattr(adapter, "fit") and not hasattr(adapter, "fit_with_validation")


def write_model(path, source, *, header=None, floats=None, target=None, truncate=0):
    raw = bytearray(Path(source).read_bytes())
    if header is not None:
        raw[20:40] = np.asarray(header, "float32").tobytes()
    if target is not None:
        L, q = np.frombuffer(bytes(raw[:8]), "int32")
        n = int(sum(np.frombuffer(bytes(raw[8:16]), "int32")))
        start = 40 + q + 4 * n
        raw[start:start + L] = target.encode()
    if floats is not None:
        raw[-4:] = np.asarray([floats], "float32").tobytes()
    path.write_bytes(bytes(raw[:len(raw) - truncate]))
    return path


@pytest.mark.parametrize("change, message", [
    ({"truncate": 4}, "size mismatch"),
    ({"header": [0.2, -1.0, 0.5, 0.0, 150.25]}, "Mean-field"),
    ({"header": [0.2, 0.01, 0.5, 0.0, 0.0]}, "positive"),
    ({"floats": float("nan")}, "Nonfinite"),
])
def test_invalid_models_fail_before_fitting(tmp_path, change, message):
    source = FIXTURE / SPEC["AMFR_HUMAN_Tsuboyama_2023_4G3O"]["model"]
    with pytest.raises(ValueError, match=message):
        evc.read_plmc_v2(write_model(tmp_path / "bad.model", source, **change))
    with pytest.raises(ValueError, match="size mismatch"):
        evc.read_plmc_v2(source, "float64")


@pytest.mark.parametrize("changes, message", [
    ({"model_sha256": "1" * 64}, "SHA256 mismatch"),
    ({"msa_start": 1}, "msa_start"),
    ({"model_id": "KCNH2_HUMAN_other"}, "model_id"),
    ({"format": "plmc_v1"}, "plmc_v2"),
    ({"provenance": {}}, "missing model provenance"),
])
def test_manifest_prerequisites_are_enforced(tmp_path, changes, message):
    path = manifest(tmp_path, KCNH2_HUMAN_Kozek_2020=changes)
    with pytest.raises(ValueError, match=message):
        evc.prepare_artifact(path, tmp_path / "out.json")
    assert not (tmp_path / "out.json").exists()


def test_coordinate_and_wild_type_mismatch_fail_preparation(tmp_path):
    assay = "KCNH2_HUMAN_Kozek_2020"
    source = FIXTURE / SPEC[assay]["model"]
    bad = write_model(tmp_path / "wrong-wt.model", source, target="VRA")
    path = manifest(tmp_path, **{assay: {"model_path": str(bad), "model_sha256": sha(bad)}})
    with pytest.raises(ValueError, match="disagrees with assay wild type"):
        evc.prepare_artifact(path, tmp_path / "out.json")
    model = evc.read_plmc_v2(source)
    with pytest.raises(ValueError, match="outside the assay"):
        evc.build_assay_fields(model, wild_type_sequence="A" * 10, msa_start=535)
    spec = json.loads(manifest(tmp_path).read_text())
    spec["dms_labels_used"] = None
    (tmp_path / "labels.json").write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="dms_labels_used"):
        evc.prepare_artifact(tmp_path / "labels.json", tmp_path / "out.json")


def test_artifact_hash_and_pins_are_required(tmp_path, artifact):
    with pytest.raises(ValueError, match="artifact_sha256"):
        evc.EVCouplingsIndependent(artifact["artifact"], "0" * 64)
    data = json.loads(Path(artifact["artifact"]).read_text())
    data["reference_sha256"] = "0" * 64
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="pins"):
        evc.EVCouplingsIndependent(tampered, sha(tampered))
    with pytest.raises(FileExistsError):
        evc.prepare_artifact(manifest(tmp_path), artifact["artifact"])


def test_proteingym_model_id_rule_matches_pinned_scorer():
    assert evc.proteingym_model_id("RASK_HUMAN_Ursu_2020", "RASK_HUMAN") == "RASK_HUMAN_Ursu_2020"
    assert evc.proteingym_model_id("X", "F7YBW7_MESOW") == "F7YBW8_MESOW"
    assert evc.proteingym_model_id("AMFR_HUMAN_Tsuboyama_2023_4G3O", "AMFR_HUMAN") == "AMFR_HUMAN"


def amfr_prepared(tmp_path, labels):
    assay = "AMFR_HUMAN_Tsuboyama_2023_4G3O"
    wt = wild_type(assay)
    source = tmp_path / "dms"
    source.mkdir(exist_ok=True)
    mutants = ["F2W", "F2Y", "Q5G", "Q5W", "F2W:Q5G", "Q3A", "Q5G:Q3A"]
    with (source / f"{assay}.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["mutant", "mutated_sequence", "DMS_score", "DMS_score_bin"])
        for mutant, label in zip(mutants, labels):
            sequence = list(wt)
            for part in mutant.split(":"):
                sequence[int(part[1:-1]) - 1] = part[-1]
            writer.writerow([mutant, "".join(sequence), label, int(label > 0)])
    return sdk.prepare(proteingym.PROTOCOL_ID, source=source, assay_ids=[assay], limit=len(mutants),
                       output=tmp_path / f"prepared-{labels[0]}")


def test_batch_runner_blocks_without_artifacts_and_scores_with_them(tmp_path, artifact):
    ident = evc.BASELINE_ID
    data = amfr_prepared(tmp_path, [0.3, -1.0, 0.8, -0.2, 0.5, 0.1, 0.0])
    blocked = run_baselines(data, output=tmp_path / "default")
    assert [r["status"] for r in blocked["baselines"]] == ["evaluated", "blocked", "blocked"]
    assert not (tmp_path / "default" / ident).exists()
    options = {ident: {"artifact": artifact["artifact"], "artifact_sha256": artifact["artifact_sha256"]}}
    run = run_baselines(data, output=tmp_path / "run", baseline_ids=[ident], baseline_options=options,
                        batch_size=2)
    record = run["baselines"][0]
    assert record["status"] == "evaluated" and record["artifact_options_supplied"] == ["artifact", "artifact_sha256"]
    # Two variants touch uncovered position 3: unscored, denominator retained.
    assert record["coverage"]["scored"] == 5 and record["completion"] == "partial"
    unscored = json.loads((tmp_path / "run" / ident / "unscored.json").read_text())
    assert set(unscored.values()) == {evc.UNSCORED_POSITION} and len(unscored) == 2
    report = json.loads((tmp_path / "run" / ident / "report.json").read_text())
    assert "No DMS labels" in report["model"]["training_overlap"]
    assert report["execution"]["adapter_provenance"]["configuration_sha256"] == artifact["artifact_sha256"]
    assert str(tmp_path) not in (tmp_path / "run" / "baseline-manifest.json").read_text()
    # Changing hidden labels changes evaluation only, never predictions.
    relabelled = amfr_prepared(tmp_path, [-5.0, 2.0, -3.0, 9.0, 1.5, -2.0, 4.0])
    run_baselines(relabelled, output=tmp_path / "relabelled", baseline_ids=[ident],
                  baseline_options=options, batch_size=7)
    assert ((tmp_path / "run" / ident / "predictions.json").read_text()
            == (tmp_path / "relabelled" / ident / "predictions.json").read_text())
    missing = copy.deepcopy(options)
    missing[ident]["artifact"] = str(tmp_path / "absent.json")
    failed = run_baselines(data, output=tmp_path / "missing", baseline_ids=[ident], baseline_options=missing)
    assert failed["baselines"][0] == {**failed["baselines"][0], "status": "failed", "error_type": "FileNotFoundError"}
    with pytest.raises(ValueError, match="baseline_options"):
        run_baselines(data, output=tmp_path / "wrong", baseline_options={"seeded-random-v1": {}})


def test_cli_prepares_artifact_and_runs_baseline(tmp_path, capsys):
    cli.main(["prepare-baseline-artifact", evc.BASELINE_ID, "--manifest", str(manifest(tmp_path)),
              "--output", str(tmp_path / "artifact.json")])
    prepared = json.loads(capsys.readouterr().out)
    amfr_prepared(tmp_path, [0.3, -1.0, 0.8, -0.2, 0.5, 0.1, 0.0])
    options = {evc.BASELINE_ID: {"artifact": prepared["artifact"], "artifact_sha256": prepared["artifact_sha256"]}}
    cli.main(["run-baselines", "--prepared", str(tmp_path / "prepared-0.3"), "--output", str(tmp_path / "run"),
              "--baseline", evc.BASELINE_ID, "--baseline-options", json.dumps(options)])
    assert json.loads(capsys.readouterr().out)["baselines"][0]["status"] == "evaluated"


def test_precision_loss_status_is_accepted_and_propagated(artifact, adapter):
    """Warning flags belong to one execution: each local state must be accepted
    and reported faithfully. The macOS receipt keeps its own historical record."""
    assert RECEIPT["assays"][R]["bfgs_warnflags"] == {"3": 2, "4": 0, "6": 2, "9": 0}
    sites = json.loads(Path(artifact["artifact"]).read_text())["assays"][R]["optimizer"]["sites"]
    for site in sites.values():
        assert site["warnflag"] in (0, 2)
        if site["warnflag"] == 2:
            assert site["status"] == "precision_loss_within_stationarity_tolerance"
            assert site["stationarity_residual"] <= evc.STATIONARITY_TOLERANCE
        else:
            assert site["status"] == "converged" and site["gradient_max_abs"] <= evc.BFGS_GTOL
    summary = adapter.provenance["optimizer"]["assays"][R]
    assert summary["precision_loss_positions"] == sorted(int(p) for p, s in sites.items() if s["warnflag"] == 2)
    assert summary["converged_sites"] == sum(s["warnflag"] == 0 for s in sites.values())
    assert summary["max_stationarity_residual"] == max(s["stationarity_residual"] for s in sites.values())


SAME_RUNTIME = bool(os.environ.get("REWIRE_EVCOUPLINGS_PYTHON") and os.environ.get("REWIRE_EVCOUPLINGS_UPSTREAM"))


@pytest.mark.skipif(not SAME_RUNTIME and not os.environ.get("REWIRE_REQUIRE_SAME_RUNTIME_UPSTREAM"),
                    reason="Same-runtime upstream parity needs REWIRE_EVCOUPLINGS_PYTHON and "
                           "REWIRE_EVCOUPLINGS_UPSTREAM; a skip is not parity evidence")
def test_same_runtime_upstream_parity(tmp_path, artifact, adapter):
    """Run the unmodified pinned upstream on this machine, then compare everything.

    Every raw field, every WT-relative contrast, every receipt score (all at
    1e-6) and the exact BFGS warning flag at every site of all three fixtures.
    The upstream environment must use this process's NumPy and SciPy versions on
    the same OS and architecture. With REWIRE_REQUIRE_SAME_RUNTIME_UPSTREAM=1 a
    missing upstream environment fails instead of skipping (used in CI).
    """
    import platform

    import scipy
    assert SAME_RUNTIME, "REWIRE_REQUIRE_SAME_RUNTIME_UPSTREAM is set but no upstream environment is configured"
    output = tmp_path / "same-runtime-receipt.json"
    subprocess.run([os.environ["REWIRE_EVCOUPLINGS_PYTHON"],
                    str(ROOT / "scripts/baseline_parity/evcouplings_upstream_receipt.py"),
                    "--upstream", os.environ["REWIRE_EVCOUPLINGS_UPSTREAM"], "--output", str(output)],
                   check=True)
    live = json.loads(output.read_text())
    environment = live["environment"]
    assert (environment["numpy"], environment["scipy"]) == (np.__version__, scipy.__version__)
    assert environment["platform"].split("-")[0] == platform.platform().split("-")[0]
    assert platform.machine() in environment["platform"]
    report, failures = adapter_report(artifact, adapter, live)
    failures += compare.warnflag_mismatches(report)
    for assay in SPEC:
        assert live["assays"][assay]["J_ij_cleared"] is True
        assert all(error and error["type"] == "ValueError" for error in live["assays"][assay]["invalid_mutants"].values())
    assert not failures, json.dumps({"failures": failures, "report": report}, indent=1)


def fake_bfgs(flag, *, x=None, objective=None, gradient=None):
    """Stand-in optimizer returning a chosen terminal state; not parity evidence."""
    def run(func, x0, fprime, args, **kw):
        xs = np.zeros_like(x0) if x is None else np.full_like(x0, x)
        f = func(xs, *args) if objective is None else objective
        g = fprime(xs, *args) if gradient is None else np.full_like(x0, gradient)
        return xs, f, g, np.eye(len(x0)), 1, 1, flag
    return run


@pytest.mark.parametrize("stub, message", [
    (fake_bfgs(1), "warnflag 1"),                       # iteration limit, finite iterate
    (fake_bfgs(3), "warnflag 3"),                       # NaN encountered
    (fake_bfgs(0), "warnflag 0"),                       # claims success at unmoved zeros
    (fake_bfgs(2), "warnflag 2"),                       # precision loss far from stationarity
    (fake_bfgs(0, objective=math.nan), "nonfinite"),
    (fake_bfgs(0, gradient=math.inf), "nonfinite"),
    (fake_bfgs(0, x=800.0), "nonfinite"),              # finite iterate, overflowing objective
])
def test_unusable_optimizer_terminal_states_block_preparation(tmp_path, monkeypatch, stub, message):
    import scipy.optimize
    monkeypatch.setattr(scipy.optimize, "fmin_bfgs", stub)
    with pytest.raises(ValueError, match=message):
        evc.prepare_artifact(manifest(tmp_path), tmp_path / "out.json")
    assert not (tmp_path / "out.json").exists()


def tamper(path, change):
    """Rewrite an artifact semantically, then hash it honestly: only meaning is wrong."""
    text = Path(path).read_text()
    if callable(change):
        data = json.loads(text)
        change(data)
        text = json.dumps(data, sort_keys=True, indent=1)
    else:
        text = text.replace(*change, 1)
    out = Path(path).with_name("tampered.json")
    out.write_text(text)
    return out, sha(out)


K = "KCNH2_HUMAN_Kozek_2020"
A = "AMFR_HUMAN_Tsuboyama_2023_4G3O"


def _set(path, value):
    def change(data):
        target = data
        for key in path[:-1]:
            target = target[key]
        if value is KeyError:
            del target[path[-1]]
        else:
            target[path[-1]] = value
    return change


def _rename_field(old, new):
    def change(data):
        fields = data["assays"][K]["fields"]
        fields[new] = fields.pop(old)
    return change


def _both_alphabets(value):
    def change(data):
        data["assays"][A]["alphabet"] = value
        data["assays"][A]["model_metadata"]["alphabet"] = value
    return change


def _negative_norms(assay, position):
    """Both norms negated: consistent ratio, but impossible as norms."""
    def change(data):
        site = data["assays"][assay]["optimizer"]["sites"][position]
        site["gradient_max_abs"] *= -1
        site["stationarity_residual"] *= -1
    return change


@pytest.mark.parametrize("change, message", [
    (_set(["dms_labels_used"], True), "dms_labels_used"),
    (_set(["dms_labels_used"], KeyError), "exactly"),
    (_set(["score_direction"], "lower_is_fitter"), "direction"),
    (_set(["assays", K, "wild_type_sequence"], "M" * 1159), "wild type"),
    (_set(["assays", K, "msa_start"], 536), "msa_start"),
    (_set(["assays", K, "model_position_offset"], 0), "offset"),
    (_set(["assays", K, "model_id"], "AMFR_HUMAN"), "model_id"),
    (_set(["assays", K, "model_metadata", "target_seq"], "VRR"), "wild type"),
    (_set(["assays", K, "model_metadata", "index_list"], [1, 2, 4]), "wild type|canonical"),
    (_rename_field("538", "536"), "canonical"),
    (_rename_field("538", "0538"), "canonical"),
    (_set(["assays", K, "fields", "539"], [0.0] * 20), "canonical"),
    (_set(["assays", A, "alphabet"], "ACDEFGHIKLMNPQRSTVYW"), "alphabet"),
    (_both_alphabets("AADEFGHIKLMNPQRSTVWY"), "alphabet"),
    (_set(["assays", K, "model_provenance", "alignment_sha256"], "not-a-digest"), "alignment_sha256"),
    (_set(["assays", K, "provenance_status"], "verified"), "provenance status"),
    (_set(["assays", K, "optimizer", "sites", "535", "warnflag"], 1), "optimizer"),
    (_set(["assays", K, "optimizer", "stationarity_tolerance"], 1.0), "optimizer"),
    # Recorded optimizer norms: nonnegative and residual bound to gradient_max_abs / N_eff.
    (_set(["assays", R, "optimizer", "sites", "3", "gradient_max_abs"], 1.0), "contradicts"),
    (_set(["assays", R, "optimizer", "sites", "3", "stationarity_residual"], 0.0), "contradicts"),
    (_set(["assays", K, "optimizer", "sites", "535", "gradient_max_abs"], -1e-9), "nonnegative"),
    (_negative_norms(R, "3"), "nonnegative"),
    # Recorded plmc_v2 header: the binary reader's integer, minimum and finite checks.
    (_set(["assays", K, "model_metadata", "L"], 3.0), "header counts"),
    (_set(["assays", A, "model_metadata", "num_symbols"], 20.0), "header counts"),
    (_set(["assays", K, "model_metadata", "N_valid"], -1), "header counts"),
    (_set(["assays", K, "model_metadata", "N_invalid"], "1"), "header counts"),
    (_set(["assays", K, "model_metadata", "num_iter"], True), "header counts"),
    (_set(["assays", K, "model_metadata", "theta"], math.nan), "finite"),
    (_set(["assays", K, "model_metadata", "lambda_J"], math.inf), "finite"),
    (_set(["assays", K, "model_metadata", "lambda_group"], None), "finite"),
    (("\"alphabet\": \"-ACDEFGHIKLMNPQRSTVY\",", "\"alphabet\": \"-ACDEFGHIKLMNPQRSTVY\", \"alphabet\": \"-ACDEFGHIKLMNPQRSTVY\","),
     "Duplicate JSON key"),
    (lambda data: data["assays"][K]["fields"]["535"].__setitem__(0, math.nan), "invalid field row"),
])
def test_rehashed_contradictory_artifacts_are_rejected(artifact, change, message):
    tampered, digest = tamper(artifact["artifact"], change)
    assert tampered.read_text() != Path(artifact["artifact"]).read_text()
    with pytest.raises(ValueError, match=message):
        evc.EVCouplingsIndependent(tampered, digest)


def test_evcouplings_notice_is_packaged_and_receipted():
    notice = proteingym.resource_path("EVcouplings-LICENSE.upstream")
    assert "Copyright (c) 2017 EVcouplings development team" in notice.read_text()
    receipt = next(r for r in json.loads(proteingym.resource_path("sources.json").read_text())
                   if r.get("local_file") == "EVcouplings-LICENSE.upstream")
    assert receipt["revision"] == evc.EVCOUPLINGS_REVISION and receipt["sha256"] == sha(notice)


def test_comparator_separates_common_offsets_from_contrast_changes(artifact, adapter):
    """Negative controls for the parity comparator itself, on a copy of the artifact."""
    assays = json.loads(Path(artifact["artifact"]).read_text())["assays"]
    predictions = {a: {m: v for m, v in RECEIPT["assays"][a]["prediction_independent"].items()} for a in SPEC}
    shifted = copy.deepcopy(assays)
    shifted[A]["fields"]["2"] = [v + 1e-3 for v in shifted[A]["fields"]["2"]]
    report, failures = compare.compare(shifted, RECEIPT["assays"], SPEC, predictions)
    site = report[A]["sites"]["2"]
    assert site["raw_symbols_outside_tolerance"] and not site["contrast_symbols_outside_tolerance"]
    assert site["mean_difference"] == pytest.approx(1e-3) and site["max_abs_difference_after_mean"] < 1e-12
    bent = copy.deepcopy(assays)
    bent[A]["fields"]["2"][bent[A]["alphabet"].index("W")] += 1e-3
    report, failures = compare.compare(bent, RECEIPT["assays"], SPEC, predictions)
    assert report[A]["sites"]["2"]["contrast_symbols_outside_tolerance"] == ["W"]
    predictions[A]["F2W"] += 1e-3
    _, failures = compare.compare(assays, RECEIPT["assays"], SPEC, predictions)
    assert any("F2W: score" in f for f in failures)


def _drop(path):
    def change(receipt):
        target = receipt
        for key in path[:-1]:
            target = target[key]
        del target[path[-1]]
    return change


@pytest.mark.parametrize("change, message", [
    (_drop(["assays", R, "independent_fields", "6"]), "receipt field sites: missing \\['6'\\]"),
    (_drop(["assays", R, "bfgs_warnflags", "6"]), "receipt warning flags: missing \\['6'\\]"),
    (_drop(["assays", A, "bfgs_warnflags"]), "receipt warning flags: missing"),
    (lambda r: r["assays"][K]["independent_fields"].__setitem__("2", [0.0] * 20), "extra \\['2'\\]"),
    (lambda r: r["assays"][K]["bfgs_warnflags"].__setitem__("3", None), "must be integers"),
    (lambda r: r["assays"][A]["independent_fields"]["5"].pop(), "one value per alphabet symbol"),
    (_drop(["assays", K, "prediction_independent", "K538F"]), "receipt scores: missing \\['K538F'\\]"),
    (_drop(["assays", K]), "assays"),
])
def test_comparator_rejects_incomplete_or_extra_receipt_coverage(artifact, change, message):
    """A receipt missing, or adding, a site, flag, row entry, score or assay cannot
    turn the all-site comparison into a partial one."""
    assays = json.loads(Path(artifact["artifact"]).read_text())["assays"]
    predictions = {a: dict(RECEIPT["assays"][a]["prediction_independent"]) for a in SPEC}
    receipt = copy.deepcopy(RECEIPT["assays"])
    compare.compare(assays, receipt, SPEC, predictions)  # complete receipt: accepted
    change({"assays": receipt})
    with pytest.raises(compare.CoverageError, match=message):
        compare.compare(assays, receipt, SPEC, predictions)


def test_comparator_rejects_incomplete_artifact_or_adapter_coverage(artifact):
    assays = json.loads(Path(artifact["artifact"]).read_text())["assays"]
    predictions = {a: dict(RECEIPT["assays"][a]["prediction_independent"]) for a in SPEC}
    partial = copy.deepcopy(assays)
    del partial[R]["optimizer"]["sites"]["6"]
    with pytest.raises(compare.CoverageError, match="artifact optimizer sites: missing \\['6'\\]"):
        compare.compare(partial, RECEIPT["assays"], SPEC, predictions)
    del predictions[R]["K6I"]
    with pytest.raises(compare.CoverageError, match="adapter scores: missing \\['K6I'\\]"):
        compare.compare(assays, RECEIPT["assays"], SPEC, predictions)
