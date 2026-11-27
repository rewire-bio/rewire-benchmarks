"""ESM C 600M masked-marginal substitution scores for ProteinGym.

Source: data repository record ``lit-protein-20261011-prime-model-esm-c`` (depth tier 2),
run guide step "Run ESMC on one sequence", taken from the model card at revision 5e366569
(spec ``data/omics/model-pages-20261127-batch8/specs/esm-c.json``, card saved as
``artifacts/esm-c/hf-card-esmc-600m.md``). Loading follows the card: ``AutoTokenizer``
and ``AutoModelForMaskedLM`` from transformers 5.16 or later, which ships ESM C. The
card's ``pip install esm`` is not needed for that path and is left out. The scoring is
the masked-marginal sum of the ESM-1b adapter, which this class extends: mask each
mutated residue in the wild-type sequence and add log P(mutant) minus log P(wild type).
The card states a context of 2048 tokens, so proteins longer than 2046 residues are left
unscored with a reason.

The checkpoint's ``config.json`` also maps the Auto classes to
``modeling_esmc_remote.py``, a shim that loads the weights through the ``esm`` package.
That path runs only with ``trust_remote_code``, which this adapter leaves off, so the
file is neither loaded nor pinned.

Weights licence: MIT, with an "other" entry pointing at the repository's third-party
notice, and use subject to the developers' acceptable use policy (card metadata and
"Out-of-Scope or Unauthorized Use Cases").

Pins: the file names and the weights digest come from the Hugging Face file listing
at the pinned revision, saved as ``resources/models/hf-trees/esmc.json``. The other
digests are of the small files fetched once at that revision; see ``pinned``.
"""

from __future__ import annotations

from typing import Any

from rewirebench.adapters.esm1b import ESM1b
from rewirebench.adapters.pinned import Checkpoint, offline_frameworks

TRANSFORMERS_VERSION = "5.16.0"
CHECKPOINT = Checkpoint(
    repository="biohub/ESMC-600M",
    revision="5e366569e675ad5ae3512f66bf0b53ffe80ee83e",
    record="lit-protein-20261011-prime-model-esm-c",
    files={
        "config.json": "2494562385804f86db4ba04cba57e4453b1c080ce9eb758443df67ee852bfd31",
        "special_tokens_map.json": "0b7245ec86c8c3aeaf61523ba70dfa79be137e6283f127bd651adc30b4f15c74",
        "tokenizer.json": "8d3447b278176e65fb3ef0224472927bf5fee3be46ea2bd77fad0111423cee1f",
        "tokenizer_config.json": "dd886af5f7a21ee11455a243e57e421bcd0289dfd15a2d2df4f5135986766a12",
        "model.safetensors": "526573c304b7718f9cfd2a8adf79638feb6752eb38056968bf21d9036b7882ce",
    },
)


class ESMC(ESM1b):
    """Pinned local checkpoint only. Construct once, then call predict in batches."""

    CHECKPOINT = CHECKPOINT
    NAME = "ESM C 600M"
    MAX_RESIDUES = 2046

    def __init__(self, checkpoint: str, *, device: str = "cpu"):
        folder = self.CHECKPOINT.verify(checkpoint)
        torch, transformers = offline_frameworks()
        if transformers.__version__ != TRANSFORMERS_VERSION:
            raise ValueError(f"This adapter is reviewed for transformers=={TRANSFORMERS_VERSION}")
        self.torch, self.device = torch, device
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            str(folder), local_files_only=True, trust_remote_code=False)
        self.model = transformers.AutoModelForMaskedLM.from_pretrained(
            str(folder), local_files_only=True, trust_remote_code=False,
            use_safetensors=True, dtype=torch.float32,
        ).eval().to(device)
        self._cache: dict[tuple[str, int], Any] = {}

    @staticmethod
    def transformers_version() -> str:
        return TRANSFORMERS_VERSION

    def provenance(self) -> dict[str, Any]:
        return {**super().provenance(), "weights_licence": "MIT plus other"}
