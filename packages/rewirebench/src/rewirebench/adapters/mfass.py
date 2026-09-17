"""Local MFASS adapters; model frameworks are imported only when requested."""

from __future__ import annotations

import hashlib
import itertools
from pathlib import Path
from typing import ClassVar

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

BASES = "ACGT"
REGIONS = ["exon", "upstr_intron", "downstr_intron"]


def kmers(k):
    return ["".join(p) for p in itertools.product(BASES, repeat=k)]


def featurise(rows, k=3, window=21):
    """Positional and compositional features. No label information is used."""
    vocab = {km: i for i, km in enumerate(kmers(k))}
    n_kmer = len(vocab)
    X = np.zeros((len(rows), 8 + 2 + len(REGIONS) + 8 + n_kmer), dtype=np.float32)

    for i, r in enumerate(rows):
        pos = float(r["rel_position"])
        i1, ex = float(r["intron1_len"]), float(r["exon_len"])
        acceptor, donor = i1, i1 + ex  # exon boundaries within the window
        c = 0
        X[i, c] = pos
        c += 1
        X[i, c] = float(r["rel_position_scaled"])
        c += 1
        X[i, c] = pos - acceptor
        c += 1  # signed dist to acceptor
        X[i, c] = pos - donor
        c += 1  # signed dist to donor
        X[i, c] = min(abs(pos - acceptor), abs(pos - donor))
        c += 1
        X[i, c] = ex
        c += 1
        X[i, c] = i1
        c += 1
        X[i, c] = float(r["intron2_len"])
        c += 1
        # Raw conservation. phyloP and phastCons are alignment statistics, not
        # trained predictors, so they belong in a trivial baseline. CADD is in the
        # cohort but is a trained model and gets its own comparator row instead.
        for col in ("phylop_score", "mean_phastCons_score"):
            v = r.get(col, "NA")
            X[i, c] = float(v) if v not in ("NA", "", None) else np.nan
            c += 1
        for reg in REGIONS:
            X[i, c] = 1.0 if r["region"] == reg else 0.0
            c += 1
        ref, alt = r["ref_allele"], r["alt_allele"]
        for b in BASES:
            X[i, c] = 1.0 if ref == b else 0.0
            c += 1
        for b in BASES:
            X[i, c] = 1.0 if alt == b else 0.0
            c += 1

        # `sequence` is reverse-complemented for 7,770 cohort rows while
        # rel_position remains in assay orientation. The validated mutant pair
        # from build_dataset is the only safe source for a centred window.
        seq = r["mutant_sequence"].upper()
        p = int(pos) - 1
        if not 0 <= p < len(seq):
            raise ValueError(f"{r['id']}: invalid assay-oriented variant position")
        lo, hi = max(0, p - window // 2), min(len(seq), p + window // 2 + 1)
        sub = seq[lo:hi]
        counts = np.zeros(n_kmer, dtype=np.float32)
        for j in range(len(sub) - k + 1):
            idx = vocab.get(sub[j : j + k])
            if idx is not None:
                counts[idx] += 1
        total = counts.sum()
        if total:
            counts /= total
        X[i, c : c + n_kmer] = counts
    return X


class KmerBaseline:
    """Corrected 21-base assay window baseline, fitted only by the evaluator."""

    capability = "scalar"
    metadata: ClassVar[dict] = {
        "method": "baseline-kmer-position-v2",
        "pretrained": False,
        "training_overlap": "fitted on supplied canonical training rows only",
    }

    def provenance(self):
        return dict(self.metadata)

    def fit(self, inputs, targets):
        if len(inputs) != len(targets) or len(set(targets)) != 2:
            raise ValueError("Training requires matching inputs and both outcome classes")
        self.model = HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.06,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=20260914,
        )
        self.model.fit(featurise(inputs), targets)
        return self

    def predict(self, inputs):
        if not hasattr(self, "model"):
            raise ValueError("Fit the baseline on canonical training rows first")
        values = self.model.predict_proba(featurise(inputs))[:, 1]
        return {row["id"]: float(value) for row, value in zip(inputs, values)}


class DNABERT2:
    """Pinned frozen DNABERT-2, loading a complete local checkpoint without network.

    Required files are listed in ARTIFACTS. The Python files are the exact code
    loaded by the archived v2 run; copy them into the checkpoint directory during
    preparation. Verifying all executable files happens before trust_remote_code.
    """

    capability = "embedding"
    MODEL_REVISION = "b5ae377faa374ee160eec1c27b8494436cc94451"
    CODE_REVISION = "7bce263b15377fc15361f52cfab88f8b586abda0"
    ARTIFACTS: ClassVar[dict] = {
        "model.safetensors": "75c91dc7bd0eea50f63e2ce6aeac2d9f20980f9efc9c715a26e4efa1e369de1c",
        "tokenizer.json": "5d178e8ce2ba55df97fff197f4b30f40133b95d7096be398c2df6b526c5d8cd3",
        "tokenizer_config.json": "f9d18c81f4dd9dd7db02e9f27cc1203228147d890bfce9167c3af6465ff5b769",
        "config.json": "ba9bdafaff0cc3e30556927474d4a179519a9864012bed2628e9f1bc23c84bfd",
        "bert_layers.py": "317ad7e9667980ac724c07f0174ba265c8446ca5230f6b473068ff1b911edcf9",
        "bert_padding.py": "44d1c68afb1f585fdc66c150d4c60f1ed44a89c006abc57d50531d71940d7421",
        "configuration_bert.py": "95fc868641b87bbcd7a32d2cd7b9f4769c27592e129daf167d14b5b8c74ec4c5",
        "flash_attn_triton.py": "568d1ac3beca0b5e1df528a1f136aa19b6489a616fcf3784f33336a50bb1de81",
    }

    def __init__(self, checkpoint, batch_size=2):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        checkpoint = Path(checkpoint).resolve()
        for name, expected in self.ARTIFACTS.items():
            path = checkpoint / name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Missing or mismatched pinned DNABERT-2 artifact: {name}")
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch, self.batch_size = torch, batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True)
        self.model = (
            AutoModel.from_pretrained(
                str(checkpoint),
                trust_remote_code=True,
                local_files_only=True,
                use_safetensors=True,
                torch_dtype=torch.float32,
            )
            .eval()
            .to("cpu")
        )
        self.metadata = {
            "method": "DNABERT-2 frozen masked-mean pair embeddings",
            "model_id": "zhihan1996/DNABERT-2-117M",
            "checkpoint_revision": self.MODEL_REVISION,
            "code_revision": self.CODE_REVISION,
            "artifact_sha256": dict(self.ARTIFACTS),
            "device": "cpu",
            "dtype": "float32",
            "pretrained": True,
            "training_overlap": "exact MFASS sequence overlap with pretraining is unreported",
        }

    def provenance(self):
        return dict(self.metadata)

    def embed(self, inputs):
        output = {}
        with self.torch.inference_mode():
            for start in range(0, len(inputs), self.batch_size):
                chunk = inputs[start : start + self.batch_size]
                sequences = [
                    row[key] for row in chunk for key in ("reference_sequence", "mutant_sequence")
                ]
                encoded = self.tokenizer(
                    sequences, return_tensors="pt", padding=True, truncation=False
                )
                states = self.model(**encoded)
                hidden = states[0] if isinstance(states, tuple) else states.last_hidden_state
                mask = encoded["attention_mask"].to(hidden.dtype).unsqueeze(-1)
                if hidden.shape[-1] != 768:
                    raise ValueError("Pinned DNABERT-2 must emit 768-dimensional states")
                means = ((hidden * mask).sum(1) / mask.sum(1).clamp(min=1)).cpu().numpy()
                means = means.reshape(len(chunk), 2, 768)
                for row, pair in zip(chunk, means):
                    output[row["id"]] = {"reference": pair[0].tolist(), "mutant": pair[1].tolist()}
        return output
