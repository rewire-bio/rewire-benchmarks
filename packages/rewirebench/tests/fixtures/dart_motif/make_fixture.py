"""Generate hand-authored MEME motifs and 350-base sequences for FIMO checks.

Nothing here is biological. Sites are planted in poly-C, which none of the
motifs match, so expected hits are known by construction. Run from this directory:

    python make_fixture.py
"""
import json
import random
from pathlib import Path

HERE = Path(__file__).parent
INDEX = {"A": 0, "C": 1, "G": 2, "T": 3}
MOTIFS = {
    "SYN_FWD": "TTGATAAGGC",   # not its own reverse complement
    "SYN_PAL": "GACGTACGTC",   # its own reverse complement
    "SYN_POLYA": "AAAAAAAA",   # narrowest motif; floods poly-A
}


def meme(motifs):
    lines = ["MEME version 4", "", "ALPHABET= ACGT", "", "strands: + -", "",
             "Background letter frequencies", "A 0.25 C 0.25 G 0.25 T 0.25", ""]
    for name, consensus in motifs.items():
        lines += [f"MOTIF {name} synthetic",
                  f"letter-probability matrix: alength= 4 w= {len(consensus)} nsites= 20 E= 0"]
        for base in consensus:
            row = [0.01] * 4
            row[INDEX[base]] = 0.97
            lines.append(" ".join(f"{v:.2f}" for v in row))
        lines.append("")
    return "\n".join(lines)


def plant(sites, length=350):
    sequence = ["C"] * length
    for start, text in sites:  # 1-based start
        sequence[start - 1:start - 1 + len(text)] = text
    return "".join(sequence)


def main():
    (HERE / "synthetic.meme").write_text(meme(MOTIFS))
    sequences = {
        "forward": plant([(101, "TTGATAAGGC")]),
        "reverse": plant([(201, "GCCTTATCAA")]),
        "palindrome": plant([(51, "GACGTACGTC")]),
        "two_sites": plant([(21, "TTGATAAGGC"), (301, "GCCTTATCAA")]),
        "none": plant([]),
        "ambiguous_site": plant([(101, "TTGANAAGGC"), (201, "TTGATAAGGC"), (340, "NNNN")]),
        "no_valid_window": ("ACGTACG" + "N") * 43 + "ACGTAC",
        "high_hit": "A" * 350,
    }
    rng = random.Random(13)
    for name in ("random_a", "random_b", "random_c"):
        sequences[name] = "".join(rng.choice("ACGT") for _ in range(350))
    sequences["random_c_masked"] = sequences["random_c"][:150] + "N" * 50 + sequences["random_c"][200:]
    assert all(len(s) == 350 for s in sequences.values())
    (HERE / "sequences.json").write_text(json.dumps(sequences, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
