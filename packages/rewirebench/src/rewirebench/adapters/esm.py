"""Offline ESM-2 8M masked-marginal example, not a published ProteinGym row.

At each mutated position mask the wild-type residue, compute log P(mutant)
minus log P(wild type), and sum across substitutions. No labels are accepted.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
from pathlib import Path
from typing import Any

from rewirebench.protocols.proteingym import AMINO_ACIDS, _substitutions

MODEL_NAME = "esm2_t6_8M_UR50D"
CHECKPOINT_SHA256 = "46f002a9870c9bdecd0ea887acb1f9a38a6b561e8f8bf8a6990b679b9d31b928"
CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t6_8M_UR50D.pt"


class ESM2Adapter:
    """Local pinned checkpoint only. Construct once, then call predict in batches.

    Sequences exceeding 1022 residues receive an explicit unscored reason.
    No silent cropping or equivalence to upstream windowing protocols is implied.
    """

    capability = "scalar"

    def __init__(self, checkpoint: str | Path, *, device: str = "cpu"):
        path = Path(checkpoint)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != CHECKPOINT_SHA256:
            raise ValueError("Checkpoint does not match the pinned ESM-2 8M weights")
        try:
            import esm
            import torch
        except ImportError as exc:
            raise ImportError("Install rewirebench[esm] before using the ESM example") from exc
        if importlib.metadata.version("fair-esm") != "2.0.0":
            raise ValueError("This adapter is reviewed for fair-esm==2.0.0")
        # The verified official checkpoint includes Python configuration objects.
        # No hub call: core loading avoids remote contact-regression downloads.
        model_data = torch.load(path, map_location="cpu", weights_only=False)
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet_core(
            MODEL_NAME, model_data, regression_data=None,
        )
        self.model.eval().to(device)
        self.device = device
        self.torch = torch
        self.converter = self.alphabet.get_batch_converter()
        self._cache: dict[tuple[str, int], object] = {}
        self.provenance = {
            "model": MODEL_NAME, "checkpoint_sha256": digest,
            "checkpoint_url": CHECKPOINT_URL, "fair_esm_version": "2.0.0",
            "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "strategy": "masked_marginals_wild_type_context_sum_log_odds",
            "maximum_sequence_length": 1022, "device": device,
            "training_overlap": "unreported; UniRef50 pretraining may overlap benchmark proteins",
            "published_result_reproduction": False,
        }

    def predict(self, inputs: list[dict]) -> dict[str, Any]:
        results = {}
        for record in inputs:
            if not {"id", "wild_type_sequence", "mutant"}.issubset(record):
                raise ValueError("ESM inputs require id, wild_type_sequence and mutant")
            if set(record) - {"id", "assay_id", "wild_type_sequence", "mutant", "mutated_sequence"}:
                raise ValueError("Unexpected input fields; labels must not reach model adapters")
            identifier, sequence = record["id"], record["wild_type_sequence"]
            if identifier in results:
                raise ValueError("Duplicate input ID")
            if (not isinstance(sequence, str) or not sequence
                    or set(sequence) - (AMINO_ACIDS | set("XBZUO"))):
                raise ValueError("ESM inputs require a nonempty valid amino-acid sequence")
            mutations = _substitutions(record["mutant"], sequence)
            if len(sequence) > 1022:
                results[identifier] = {
                    "score": None,
                    "reason": "ESM example supports at most 1022 residues; windowed inference not implemented",
                }
                continue
            score = 0.0
            for index, wild_type, mutant in mutations:
                cache_key = (sequence, index)
                if cache_key not in self._cache:
                    _, _, tokens = self.converter([("protein", sequence)])
                    tokens[0, index + 1] = self.alphabet.mask_idx
                    with self.torch.inference_mode():
                        logits = self.model(tokens.to(self.device))["logits"]
                        self._cache[cache_key] = self.torch.log_softmax(
                            logits[0, index + 1], dim=-1,
                        ).cpu()
                probabilities = self._cache[cache_key]
                score += float(probabilities[self.alphabet.get_idx(mutant)]
                               - probabilities[self.alphabet.get_idx(wild_type)])
            results[identifier] = score
        return results
