"""Small CPU learning control, not a FLIP2 published model.

For PDZ3, count each colon-separated component separately, retaining the empty
partner as a zero vector. Other datasets use a zero-filled second component.
The protocol fits the fixed ridge head; this adapter never sees labels.
"""


from typing import ClassVar


class CompositionEmbeddings:
    provenance: ClassVar[dict] = {
        "implementation": "rewire-flip2-composition-v1",
        "role": "local learning control; not the published one-hot baseline",
        "training_overlap": "No pretraining; protocol trains the regression head",
    }

    def embed(self, inputs):
        alphabet = "ACDEFGHIKLMNPQRSTVWY"
        output = {}
        for row in inputs:
            parts = row["sequence"].split(":")
            if len(parts) == 1:
                parts.append("")
            if len(parts) != 2:
                raise ValueError("Expected one sequence or a colon-delimited pair")
            output[row["id"]] = [
                part.count(letter) / max(len(part), 1)
                for part in parts for letter in alphabet
            ]
        return output
