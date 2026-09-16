"""MFASS-v2: frozen DNABERT-2 pair embeddings plus a train-only logistic head.

Protocol is fixed before held-out scoring: use the 170bp assay-oriented reference
and mutant sequence, masked mean of last_hidden_state (never the randomly
initialized pooler), concatenate reference and mutant-minus-reference vectors,
fit StandardScaler and balanced L2 logistic regression (C=0.1) only on split-v2
train rows, and score all test rows once. No validation search or test tuning.
"""
from __future__ import annotations

import argparse
import csv
import inspect
import json
import os
import pathlib
import shutil
import time

import numpy as np
from rewirebench import metrics as M
from rewirebench.results import BenchmarkResult, write_result
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .run_dnabert2_pilot import MIN_FREE_BYTES, MODEL_ID, MODEL_REVISION, sha256
from .validation import validate_canonical_split

METHOD = "dnabert2-117m-frozen-pair-logreg"
TOTAL_LIMIT_HOURS = 12
HEAD_C = 0.1
HEAD_SEED = 20260914


def masked_mean(hidden, attention_mask):
    """Pool pretrained token states, excluding tokenizer padding."""
    mask = attention_mask.to(hidden.dtype).unsqueeze(-1)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


def load_rows(cohort, split):
    validate_canonical_split(split)
    with open(cohort, newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    with open(split, newline="") as fh:
        sp = {r["id"]: (r["group"], r["split"]) for r in csv.DictReader(fh, delimiter="\t")}
    if len(rows) != len(sp) or {r["id"] for r in rows} != set(sp):
        raise ValueError("cohort and split IDs do not reconcile")
    if len(rows) != 27733:
        raise ValueError("MFASS-v2 expects the published 27,733-variant cohort")
    for r in rows:
        p = int(r["rel_position"]) - 1
        ref, mut = r["reference_sequence"], r["mutant_sequence"]
        if not (len(ref) == len(mut) == 170 and 0 <= p < 170):
            raise ValueError(f"{r['id']}: invalid assay pair")
        if [i for i, (a, b) in enumerate(zip(ref, mut)) if a != b] != [p]:
            raise ValueError(f"{r['id']}: mutant is not at rel_position - 1")
    return rows, sp


def embed_pairs(rows, tokenizer, model, torch, batch_size):
    vectors = np.empty((len(rows), 1536), dtype=np.float32)
    token_lengths = []
    t0 = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            if time.perf_counter() - t0 >= TOTAL_LIMIT_HOURS * 3600:
                raise RuntimeError("12-hour local run cap reached before held-out scoring")
            chunk = rows[start:start + batch_size]
            sequences = [s for r in chunk for s in
                         (r["reference_sequence"], r["mutant_sequence"])]
            encoded = tokenizer(sequences, return_tensors="pt", padding=True,
                                truncation=False)
            mask = encoded["attention_mask"]
            token_lengths.extend(int(x) for x in mask.sum(dim=1))
            output = model(**encoded)
            hidden = output[0] if isinstance(output, tuple) else output.last_hidden_state
            if hidden.shape[:2] != mask.shape or hidden.shape[-1] != 768:
                raise ValueError("DNABERT-2 token output did not match the declared 768d protocol")
            means = masked_mean(hidden, mask).cpu().numpy().reshape(len(chunk), 2, 768)
            vectors[start:start + len(chunk), :768] = means[:, 0, :]
            vectors[start:start + len(chunk), 768:] = means[:, 1, :] - means[:, 0, :]
            if (start + len(chunk)) % 2000 < batch_size:
                elapsed = time.perf_counter() - t0
                print(f"embedded {start + len(chunk)}/{len(rows)} pairs, "
                      f"{elapsed / (start + len(chunk)):.4f}s/pair", flush=True)
    return vectors, time.perf_counter() - t0, token_lengths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="benchmarks/mfass/data/cohort.tsv")
    ap.add_argument("--split", default="benchmarks/mfass/splits/split-v2.tsv")
    ap.add_argument("--pilot", default="benchmarks/mfass/feasibility/dnabert2-local.json")
    ap.add_argument("--out", default=f"benchmarks/mfass/results/{METHOD}.json")
    ap.add_argument("--batch-size", type=int, default=2)
    args = ap.parse_args()
    if args.batch_size < 1:
        ap.error("batch-size must be positive")

    rows, sp = load_rows(args.cohort, args.split)
    pilot = json.loads(pathlib.Path(args.pilot).read_text())
    if (pilot["status"] != "feasible" or pilot["model"] != MODEL_ID or
            pilot["checkpoint_and_code_revision"] != MODEL_REVISION or
            pilot["source"]["cohort_sha256"] != sha256(pathlib.Path(args.cohort)) or
            pilot["source"]["split_sha256"] != sha256(pathlib.Path(args.split))):
        raise SystemExit("Pinned feasible pilot does not match this dataset/model")
    projected_hours = pilot["timing_seconds"]["embed_pairs"] / pilot["source"]["pairs"] * len(rows) / 3600
    if projected_hours >= TOTAL_LIMIT_HOURS:
        raise SystemExit(f"Projected full embedding time {projected_hours:.2f}h exceeds 12h")
    if shutil.disk_usage(pathlib.Path(args.out).parent).free < MIN_FREE_BYTES:
        raise SystemExit("Less than 1 GiB free disk; full run refused")
    print(f"Full cohort projected embedding {projected_hours:.2f}h; disk reserve passes", flush=True)

    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoModel, AutoTokenizer

    torch.set_num_threads(os.cpu_count() or 4)
    t_all = time.perf_counter()
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModel.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True,
        use_safetensors=True, torch_dtype=torch.float32,
    ).eval().to("cpu")
    t_load = time.perf_counter() - t0
    code_path = pathlib.Path(inspect.getfile(type(model)))
    code_module = type(model).__module__
    weights_path = pathlib.Path(hf_hub_download(MODEL_ID, "model.safetensors",
                                                 revision=MODEL_REVISION))
    tokenizer_path = pathlib.Path(hf_hub_download(MODEL_ID, "tokenizer.json",
                                                   revision=MODEL_REVISION))
    checkpoint_hash = sha256(weights_path)
    pinned_code_path = pathlib.Path(hf_hub_download(MODEL_ID, "bert_layers.py",
                                                   revision=MODEL_REVISION))
    if sha256(code_path) != sha256(pinned_code_path):
        raise RuntimeError("Loaded remote code differs from the pinned repository revision")
    if (sha256(code_path) != pilot["environment"].get("loaded_remote_code_sha256") or
            checkpoint_hash != pilot["environment"].get("checkpoint_safetensors_sha256")):
        raise RuntimeError("Loaded code/checkpoint differs from the verified pilot; rerun the pilot")

    vectors, t_embed, token_lengths = embed_pairs(rows, tokenizer, model, torch, args.batch_size)
    # The encoder is frozen; discard it before fitting the small supervised head.
    del model
    train_idx = np.array([i for i, r in enumerate(rows) if sp[r["id"]][1] == "train"])
    test_idx = np.array([i for i, r in enumerate(rows) if sp[r["id"]][1] == "test"])
    train = [rows[i] for i in train_idx]
    test = [rows[i] for i in test_idx]
    ytr = np.array([int(r["sdv"]) for r in train])
    yte = np.array([int(r["sdv"]) for r in test])
    if len(train) != 19409 or len(test) != 8324 or ytr.sum() != 735 or yte.sum() != 315:
        raise ValueError("Canonical split counts changed; held-out result refused")

    t0 = time.perf_counter()
    scaler = StandardScaler().fit(vectors[train_idx])
    xtr = scaler.transform(vectors[train_idx])
    clf = LogisticRegression(C=HEAD_C, class_weight="balanced", solver="liblinear",
                             max_iter=1000, random_state=HEAD_SEED)
    clf.fit(xtr, ytr)
    t_head = time.perf_counter() - t0
    t0 = time.perf_counter()
    scores = clf.predict_proba(scaler.transform(vectors[test_idx]))[:, 1]
    t_predict = time.perf_counter() - t0
    if not np.all(np.isfinite(scores)):
        raise ValueError("Nonfinite test scores; result refused")

    out = pathlib.Path(args.out)
    if shutil.disk_usage(out.parent).free < MIN_FREE_BYTES:
        raise SystemExit("Disk reserve fell below 1 GiB; result refused")
    groups = [sp[r["id"]][0] for r in test]
    result = BenchmarkResult(
        benchmark="mfass-v2", method=METHOD, family="pretrained encoder",
        description=("Frozen DNABERT-2 117M masked-mean reference/mutant embeddings, "
                     "train-only balanced L2 logistic head"),
        split=args.split,
        metrics=M.point_metrics(yte, scores, capacity=100),
        coverage={"scored": len(test), "unscored": 0, "denominator": len(test)},
        timing_seconds={
            "load_tokenizer_and_checkpoint": round(t_load, 3),
            "embed_all_pairs": round(t_embed, 3),
            "fit_head_train_only": round(t_head, 3),
            "predict_test": round(t_predict, 3),
            "end_to_end": round(time.perf_counter() - t_all, 3),
            "per_variant_total": round((t_embed + t_head + t_predict + t_load) / len(rows), 6),
        },
        independent_groups=len(set(groups)), pretrained=True,
        contamination=("DNABERT-2 pretraining used multi-species genome sequences. MFASS assay "
                       "outcomes were not a declared pretraining target; exact sequence or exon "
                       "overlap with pretraining has not been checked."),
        config={
            "model_id": MODEL_ID, "checkpoint_revision": MODEL_REVISION,
            "checkpoint_safetensors_sha256": checkpoint_hash,
            "remote_code_module": code_module,
            "remote_code_bert_layers_sha256": sha256(code_path),
            "tokenizer_json_sha256": sha256(tokenizer_path),
            "cohort_sha256": sha256(pathlib.Path(args.cohort)),
            "split_sha256": sha256(pathlib.Path(args.split)),
            "context_bases": 170, "token_length_min": min(token_lengths),
            "token_length_max": max(token_lengths),
            "pooling": "attention-mask mean of last_hidden_state; random pooler ignored",
            "pair_features": "concat(reference_mean, mutant_mean - reference_mean)",
            "device": "cpu", "precision": "float32", "batch_size_pairs": args.batch_size,
            "head": "StandardScaler + balanced L2 LogisticRegression",
            "head_C": HEAD_C, "head_solver": "liblinear", "head_max_iter": 1000,
            "head_iterations": int(clf.n_iter_[0]), "seed": HEAD_SEED,
            "trained_on_variants": len(train), "trained_on_positives": int(ytr.sum()),
            "heldout_variants": len(test), "pilot_projected_full_cohort_hours": projected_hours,
        },
        notes=("This is a supervised assay-specific head on frozen pretrained embeddings; "
               "the encoder itself was not fine-tuned. Assay-oriented source pairs were "
               "validated before any label was used."),
    )
    write_result(result, out)
    np.save(out.with_suffix(".scores.npy"), scores)
    np.savez_compressed(out.with_suffix(".head.npz"),
                        scaler_mean=scaler.mean_, scaler_scale=scaler.scale_,
                        coefficients=clf.coef_, intercept=clf.intercept_)
    with out.with_suffix(".predictions.tsv").open("w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
        writer.writerow(["id", "group", "label", "score"])
        for r, s in zip(test, scores):
            writer.writerow([r["id"], sp[r["id"]][0], r["sdv"], f"{s:.10f}"])
    print(json.dumps({"method": METHOD, "metrics": result.metrics,
                      "coverage": result.coverage, "timing_seconds": result.timing_seconds,
                      "head_iterations": int(clf.n_iter_[0])}, indent=2), flush=True)


if __name__ == "__main__":
    main()
