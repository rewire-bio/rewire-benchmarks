"""Identity gates for the predeclared MFASS-v2 protocol."""
import hashlib
from pathlib import Path

CANONICAL_SPLIT_SHA256 = "999ebcb7e63a5c5eaa8780fa468e59ac1f934260ad50102814174c396317f052"


def validate_canonical_split(path):
    """Refuse altered assignments even if cohort/class counts are unchanged.

    This fixed protocol uses the checked-in split-v2.tsv, whose whole exon/gene
    components were assigned before model fitting. A different split needs a new
    protocol identifier, rather than silently overwriting MFASS-v2 evidence.
    """
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest != CANONICAL_SPLIT_SHA256:
        raise ValueError("MFASS-v2 requires the unchanged canonical split-v2.tsv (SHA-256 mismatch)")
    return digest
