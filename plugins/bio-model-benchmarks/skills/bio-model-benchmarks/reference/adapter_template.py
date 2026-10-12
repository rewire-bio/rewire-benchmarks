"""Template rewirebench adapter. Copy it next to your model code and replace the marked parts.

Run it with:

    rewirebench run --prepared PREPARED --output NEW_DIR \
      --adapter my_adapter:MyAdapter --adapter-options '{"checkpoint": "/path/to/weights"}' \
      --model-name 'My model, configuration X' --training-overlap 'unreported'

`--adapter` is module:class, so the directory holding this file must be importable
(run from that directory or put it on PYTHONPATH). Adapter options become keyword
arguments to __init__.

What the adapter sees, by protocol (run `rewirebench inspect PROTOCOL` for the exact contract):

- mfass-v2: id, reference_sequence, mutant_sequence (validated, assay oriented), ref_allele,
  alt_allele, rel_position, rel_position_scaled, region, strand, exon_len, intron1_len,
  intron2_len, phylop_score, mean_phastCons_score. Higher score = more splice disruption.
- proteingym-v1.3-dms-substitutions: id, assay_id, wild_type_sequence, mutated_sequence, mutant.
  Higher score = higher fitness. Zero-shot: an adapter with fit() is refused.
- flip2-fitness-v1, mrnabench-sample-mrl-v1, dart-eval-task1-zero-shot-v1: id, sequence.
  DART is zero-shot (no fit). PDZ3 sequences are "protein:peptide".
- genomic-benchmarks-v2: id, sequence. Return a class index, or a probability for binary tasks.
- tdc-admet-group-v1: id, smiles. Return the property, or a positive-class score.

Labels are never passed to predict(). Test labels are never passed at all.
"""

from __future__ import annotations

from typing import ClassVar


class MyAdapter:
    # Recorded in report.json. Revisions are 40 lowercase hex characters, SHA-256 values 64;
    # only these named fields are exported, so do not put private paths in them.
    provenance: ClassVar[dict] = {
        "method": "REPLACE: short method name",
        "pretrained": True,
        "training_overlap": "unreported",
        # "checkpoint_revision": "0123...",  # optional
        # "weights_sha256": "abcd...",       # optional
    }

    def __init__(self, checkpoint: str | None = None, device: str = "cpu"):
        # REPLACE: load your model here. Loading time is excluded from inference timing.
        self.checkpoint = checkpoint
        self.device = device
        self.model = None

    # Optional, only where `rewirebench inspect` shows allow_fit: true. Receives the
    # training partition only. Delete this method for zero-shot protocols.
    # def fit(self, train_inputs, train_targets):
    #     ...

    def predict(self, inputs):
        """Return {id: finite float}. Never drop an id silently."""
        scores = {}
        for row in inputs:
            try:
                scores[row["id"]] = float(self.score(row))
            except ValueError as error:
                # An unscored input stays in the denominator; say why.
                scores[row["id"]] = {"score": None, "reason": f"unsupported input: {error}"}
        return scores

    def score(self, row) -> float:
        # REPLACE: call your model. This placeholder counts GC in the main sequence field,
        # which is only useful to check that the plumbing works.
        sequence = row.get("mutant_sequence") or row.get("mutated_sequence") or row.get("sequence")
        if not sequence:
            raise ValueError("no sequence field")
        sequence = sequence.upper()
        return (sequence.count("G") + sequence.count("C")) / len(sequence)


class MyEncoder:
    """Frozen-embedding variant. Use with --prediction-type embedding.

    mfass-v2-frozen-encoder expects {id: {"reference": vector, "mutant": vector}};
    flip2-fitness-v1 and mrnabench-sample-mrl-v1 expect {id: vector}. The protocol fits its
    own fixed head on training embeddings only.
    """

    provenance: ClassVar[dict] = {"method": "REPLACE: encoder name and pooling",
                                  "pretrained": True, "training_overlap": "unreported"}

    def embed(self, inputs):
        # REPLACE: return one finite 1D vector per id, all of the same length.
        return {row["id"]: [float(len(row["sequence"]))] for row in inputs}
