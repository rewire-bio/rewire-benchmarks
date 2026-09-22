"""Frozen ESM-2 8M/35M mean-pooled representations for protocol-owned probes.

Pooling follows the official ESM representation example: final-layer residue
representations, excluding BOS/EOS and padding. No labels enter the encoder.
This is a new evaluated pipeline, not reproduction of a published model score.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from importlib.resources import files
from pathlib import Path
from types import SimpleNamespace

from rewirebench.adapters.esm import ESM2Adapter
from rewirebench.protocols.proteingym import AMINO_ACIDS


class ESM2Embeddings:
    def __init__(self, checkpoint, *, device="cpu", model_name="esm2_t6_8M_UR50D"):
        manifest = json.loads(files("rewirebench").joinpath(
            "resources/models/esm2-embeddings.json").read_text())
        if model_name not in manifest["models"]:
            raise ValueError("Unsupported ESM-2 embedding checkpoint identity")
        specification = manifest["models"][model_name]
        path = Path(checkpoint)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != specification["checkpoint_sha256"]:
            raise ValueError("Checkpoint does not match the pinned ESM-2 embedding weights")
        if model_name == "esm2_t6_8M_UR50D":
            self.encoder = ESM2Adapter(path, device=device)
        else:
            import esm
            import torch

            if importlib.metadata.version("fair-esm") != "2.0.0":
                raise ValueError("This adapter requires fair-esm==2.0.0")
            # Verify bytes before loading the official pickle-based checkpoint.
            data = torch.load(path, map_location="cpu", weights_only=False)
            model, alphabet = esm.pretrained.load_model_and_alphabet_core(
                model_name, data, regression_data=None)
            model.eval().to(device)
            self.encoder = SimpleNamespace(model=model, alphabet=alphabet,
                converter=alphabet.get_batch_converter(), torch=torch, device=device)
        self.layer = specification["layer"]
        self.provenance = {
            "model": model_name, "checkpoint_sha256": digest,
            "checkpoint_url": specification["checkpoint_url"],
            "fair_esm_version": "2.0.0", "device": device,
            "maximum_sequence_length": 1022,
            "training_overlap": "unreported; UniRef50 pretraining may overlap benchmark proteins",
            "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "strategy": "frozen_final_layer_residue_mean_excluding_special_tokens",
            "representation_layer": self.layer,
            "reference": "https://github.com/facebookresearch/esm/blob/" + manifest["upstream_revision"] + "/README.md#usage",
            "published_result_reproduction": False,
        }

    def embed(self, inputs):
        ids = [row.get("id") for row in inputs]
        if len(ids) != len(set(ids)) or any(not isinstance(x, str) or not x for x in ids):
            raise ValueError("Embedding inputs require unique nonempty IDs")
        for row in inputs:
            if set(row) != {"id", "sequence"}:
                raise ValueError("ESM embedding inputs allow only id and sequence, never labels")
            sequence = row["sequence"]
            if (not isinstance(sequence, str) or not 1 <= len(sequence) <= 1022
                    or set(sequence) - (AMINO_ACIDS | set("XBZUO"))):
                raise ValueError("ESM embeddings require 1..1022 valid amino-acid residues; no cropping")
        if not inputs:
            return {}
        encoder = self.encoder
        _, _, tokens = encoder.converter([(row["id"], row["sequence"]) for row in inputs])
        with encoder.torch.inference_mode():
            vectors = encoder.model(tokens.to(encoder.device), repr_layers=[self.layer],
                                    return_contacts=False)["representations"][self.layer]
        return {row["id"]: vectors[i, 1:len(row["sequence"]) + 1].mean(0).cpu().tolist()
                for i, row in enumerate(inputs)}
