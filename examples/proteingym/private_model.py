"""Replace this class with your own model; never return DMS labels as predictions."""


class PrivateModel:
    """Minimal callable interface; model training is separate from this zero-shot track."""

    def __init__(self, model):
        self.model = model
        self.provenance = {"training_overlap": "unreported", "method": "user supplied private model"}

    def predict(self, inputs):
        # inputs contain id, assay_id, wild_type_sequence, mutated_sequence, mutant.
        # The evaluator never supplies DMS_score, DMS_score_bin or target here.
        return {
            row["id"]: float(self.model.fitness_delta(
                wild_type=row["wild_type_sequence"], mutant=row["mutated_sequence"],
            ))
            for row in inputs
        }
