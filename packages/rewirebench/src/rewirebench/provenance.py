"""Public provenance names shared by export and submission validation.

Both keys and values must be checked: a checksum value does not make a private
path or customer name in its dictionary key safe to disclose.
"""

ADAPTER_FIELDS = {
    "checkpoint_revision": 40,
    "code_revision": 40,
    "checkpoint_sha256": 64,
    "implementation_sha256": 64,
    "weights_sha256": 64,
    "configuration_sha256": 64,
}

PUBLIC_FIELDS = {
    "source_sha256": 64,
    "source_csv_sha256": 64,
    "cohort_sha256": 64,
    "split_sha256": 64,
    "raw_sha256": 64,
    "annotation_sha256": 64,
    "reference_sha256": 64,
    "train_sha256": 64,
    "test_sha256": 64,
    "validation_sha256": 64,
    "dataset_manifest_sha256": 64,
    "evaluator_sha256": 64,
    "train_val_sha256": 64,
    "sdk_code_sha256": 64,
    "sif_sha256": 64,
    "oci_image_sha256": 64,
    "evidence_manifest_sha256": 64,
    "upstream_revision": 40,
    "runner_revision": 40,
    "protocol_revision": 40,
    "code_revision": 40,
    "source_revision": 40,
    "dataset_revision": 40,
    **{"model_" + key: length for key, length in ADAPTER_FIELDS.items()},
}


def is_digest(value, length):
    return (
        isinstance(value, str)
        and len(value) == length
        and all(char in "0123456789abcdef" for char in value)
    )
