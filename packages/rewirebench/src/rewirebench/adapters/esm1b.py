"""ESM-1b masked-marginal substitution scores for ProteinGym, from the transformers port.

Source: data repository record ``uc20260930-model-esm-1b`` (depth tier 2), spec
``data/omics/model-pages-20261126-batch5/specs/esm-1b.json``. The scoring is the one
the existing ESM-2 adapter uses (``rewirebench.adapters.esm``): at each mutated position
mask the wild-type residue in the wild-type sequence, take log P(mutant) minus
log P(wild type) from the logits at that position, and sum over the substitutions.

Difference from the run guide: its steps install ``fair-esm`` and call
``esm.pretrained.esm1b_t33_650M_UR50S()``, which downloads an unversioned file. The
record also holds the transformers port ``biohub/esm1b_t33_650M_UR50S`` with its card
and ``config.json`` saved at revision 7b37824b (``artifacts/esm-1b/hf-card.md``,
``hf-config.json``). This adapter loads that port with ``EsmForMaskedLM`` and
``EsmTokenizer`` because only it has a pinned revision. The card says the inputs are
``<cls>`` followed by the sequence, and that training cropped sequences to 1023 tokens
after ``<cls>``; sequences longer than 1022 residues are left unscored with a reason,
as in the ESM-2 adapter.

The weights are ``pytorch_model.bin``, a pickle, unpickled after the digest check.

Pins: the file names and the weights digest come from the Hugging Face file listing
at the pinned revision, saved as ``resources/models/hf-trees/esm1b.json``. The other
digests are of the small files fetched once at that revision; see ``pinned``.
"""

from __future__ import annotations

from typing import Any

from rewirebench.adapters.pinned import (
    Checkpoint,
    as_array,
    log_softmax,
    offline_frameworks,
    to_device,
    unique_ids,
)
from rewirebench.protocols.proteingym import AMINO_ACIDS, _substitutions

TRANSFORMERS_VERSION = "4.41.1"
MAX_RESIDUES = 1022
FIELDS = {"id", "assay_id", "wild_type_sequence", "mutant", "mutated_sequence"}
CHECKPOINT = Checkpoint(
    repository="biohub/esm1b_t33_650M_UR50S",
    revision="7b37824baec4d3658e1df7479222a7c79b465b76",
    record="uc20260930-model-esm-1b",
    files={
        "config.json": "e6296f876d44faeb9f4d206386af6db5e834763b14eb9a1c2539a3fb9a43edf5",
        "pytorch_model.bin": "ebb8a35768179581f91fc01b93cf78facba2821c3bab1452abb38ef2a7f6e7b1",
        "vocab.txt": "0b82cc0a7c7cf9e567b1e5892d793285b9fbae822c964ca48696f7db44598e03",
        "tokenizer_config.json": "7e9161ecdb548ec45a41cbc6b24aa4476fdd418461f491c4207baa99419a29ad",
        "special_tokens_map.json": "3aedcd4211c0d43aec4e607ff60a63255f3174ead795e997350f09a5f8cd9ee1",
    },
)


def check_record(record: dict[str, Any]) -> tuple[str, str, list[tuple[int, str, str]]]:
    """The ID, wild-type sequence and substitutions of one ProteinGym input."""
    if not {"id", "wild_type_sequence", "mutant"} <= set(record) or set(record) - FIELDS:
        raise ValueError("Inputs need id, wild_type_sequence and mutant, and never labels")
    sequence = record["wild_type_sequence"]
    if (not isinstance(sequence, str) or not sequence
            or set(sequence) - (AMINO_ACIDS | set("XBZUO"))):
        raise ValueError("Inputs need a nonempty valid amino-acid sequence")
    substitutions = _substitutions(record["mutant"], sequence)
    mutated = record.get("mutated_sequence")
    if mutated is not None:
        expected = list(sequence)
        for index, _, alt in substitutions:
            expected[index] = alt
        if mutated != "".join(expected):
            raise ValueError(f"{record['id']}: mutated_sequence does not match mutant")
    return record["id"], sequence, substitutions


class ESM1b:
    """Pinned local checkpoint only. Construct once, then call predict in batches."""

    CHECKPOINT = CHECKPOINT
    NAME = "ESM-1b"
    MAX_RESIDUES = MAX_RESIDUES

    def __init__(self, checkpoint: str, *, device: str = "cpu"):
        folder = self.CHECKPOINT.verify(checkpoint)
        torch, transformers = offline_frameworks()
        if transformers.__version__ != TRANSFORMERS_VERSION:
            raise ValueError(f"This adapter is reviewed for transformers=={TRANSFORMERS_VERSION}")
        self.torch, self.device = torch, device
        self.tokenizer = transformers.EsmTokenizer.from_pretrained(
            str(folder), local_files_only=True)
        self.model = transformers.EsmForMaskedLM.from_pretrained(
            str(folder), local_files_only=True, torch_dtype=torch.float32,
        ).eval().to(device)
        self._cache: dict[tuple[str, int], Any] = {}

    def provenance(self) -> dict[str, Any]:
        return {
            **self.CHECKPOINT.provenance(),
            "method": f"{self.NAME} masked marginals, wild-type context, summed log-odds",
            "transformers_version": self.transformers_version(),
            "maximum_sequence_length": self.MAX_RESIDUES, "device": self.device,
            "training_overlap": "unreported; UniRef50 pretraining may overlap benchmark proteins",
            "published_result_reproduction": False,
        }

    @staticmethod
    def transformers_version() -> str:
        return TRANSFORMERS_VERSION

    def _log_probabilities(self, sequence: str, index: int) -> Any:
        """Log-probabilities over the vocabulary at one masked residue."""
        key = (sequence, index)
        if key not in self._cache:
            encoded = dict(self.tokenizer(sequence, return_tensors="pt"))
            ids = encoded["input_ids"]
            masked = ids.clone() if hasattr(ids, "clone") else ids.copy()
            masked[0, index + 1] = self.tokenizer.mask_token_id
            encoded["input_ids"] = masked
            with self.torch.inference_mode():
                logits = self.model(**{k: to_device(v, self.device)
                                       for k, v in encoded.items()})[0]
            self._cache[key] = log_softmax(as_array(logits)[0, index + 1])
        return self._cache[key]

    def predict(self, inputs: list[dict[str, Any]]) -> dict[str, Any]:
        unique_ids(inputs)
        checked = [check_record(record) for record in inputs]
        results: dict[str, Any] = {}
        for identifier, sequence, substitutions in checked:
            if len(sequence) > self.MAX_RESIDUES:
                results[identifier] = {
                    "score": None,
                    "reason": f"{self.NAME} scores at most {self.MAX_RESIDUES} residues; "
                              "windowed inference is not implemented",
                }
                continue
            score = 0.0
            for index, wild_type, mutant in substitutions:
                log_p = self._log_probabilities(sequence, index)
                token = self.tokenizer.convert_tokens_to_ids
                score += float(log_p[token(mutant)] - log_p[token(wild_type)])
            results[identifier] = score
        return results
