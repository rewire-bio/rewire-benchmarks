"""The assay coordinate must not be applied to a reversed sequence."""
from pathlib import Path

import numpy as np
import pytest
from mfass.build_dataset import assay_pair, build
from mfass.run_baseline import featurise

SOURCE = Path("benchmarks/mfass/data/snv_data_clean.txt")


def source_row(strand="+", reversed_legacy=False):
    ref = "A" * 20 + "C" + "A" * 149
    mutant = "A" * 20 + "T" + "A" * 149
    legacy = mutant.translate(str.maketrans("ACGT", "TGCA"))[::-1] if reversed_legacy else mutant
    return {
        "id": "synthetic", "strand": strand, "rel_position": "21",
        "ref_allele": "C" if strand == "+" else "G",
        "alt_allele": "T" if strand == "+" else "A",
        "natural_seq": ref, "original_seq": mutant, "sequence": legacy,
    }


@pytest.mark.parametrize("strand", ["+", "-"])
@pytest.mark.parametrize("reversed_legacy", [False, True])
def test_assay_pair_and_legacy_orientation(strand, reversed_legacy):
    row = source_row(strand, reversed_legacy)
    reference, mutant, orientation = assay_pair(row)
    assert reference[20] == "C" and mutant[20] == "T"
    assert orientation == ("reverse_complement" if reversed_legacy else "assay")


def test_assay_pair_refuses_mislocated_mutation():
    row = source_row()
    row["rel_position"] = "22"
    with pytest.raises(ValueError, match="rel_position"):
        assay_pair(row)


def test_baseline_uses_validated_mutant_not_legacy_sequence():
    row = source_row(reversed_legacy=True)
    common = {
        "id": row["id"], "rel_position": row["rel_position"],
        "rel_position_scaled": "0.2", "intron1_len": "10", "exon_len": "100",
        "intron2_len": "60", "phylop_score": "NA", "mean_phastCons_score": "NA",
        "label": "exon", "ref_allele": row["ref_allele"],
        "alt_allele": row["alt_allele"], "mutant_sequence": row["original_seq"],
    }
    forward = {**common, "sequence": row["original_seq"]}
    reversed_row = {**common, "sequence": row["sequence"]}
    np.testing.assert_array_equal(featurise([forward]), featurise([reversed_row]))


@pytest.mark.skipif(not SOURCE.exists(), reason="published source table is downloaded separately")
def test_published_cohort_has_exact_reverse_complement_count():
    cohort, _ = build(SOURCE)
    assert len(cohort) == 27733
    assert sum(r["legacy_sequence_orientation"] == "reverse_complement" for r in cohort) == 7770
    assert sum(r["legacy_sequence_orientation"] == "assay" for r in cohort) == 19963
    assert all(r["reference_sequence"][int(r["rel_position"]) - 1] !=
               r["mutant_sequence"][int(r["rel_position"]) - 1] for r in cohort)
