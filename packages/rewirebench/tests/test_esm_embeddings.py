import pytest
from rewirebench.adapters.esm_embeddings import ESM2Embeddings


@pytest.mark.parametrize("inputs", [
    [{"id": "a", "sequence": "ACD", "label": 1}],
    [{"id": "a", "sequence": "A" * 1023}],
    [{"id": "a", "sequence": "ACD"}, {"id": "a", "sequence": "ACD"}],
    [{"id": "a", "sequence": "A?"}],
])
def test_input_validation_precedes_model_access(inputs):
    adapter = object.__new__(ESM2Embeddings)
    with pytest.raises(ValueError):
        adapter.embed(inputs)


@pytest.mark.parametrize("layer", [6, 12])
def test_pooling_excludes_special_tokens_and_padding(layer):
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace
    def forward(tokens, **kwargs):
        assert kwargs == {"repr_layers": [layer], "return_contacts": False}
        return {"representations": {layer: torch.tensor([
            [[999.], [1.], [3.], [999.], [999.]],
            [[999.], [2.], [4.], [6.], [999.]],
        ])}}
    adapter = object.__new__(ESM2Embeddings)
    adapter.layer = layer
    adapter.encoder = SimpleNamespace(torch=torch, device="cpu", model=forward,
        converter=lambda _: (None, None, torch.zeros((2, 5), dtype=torch.long)))
    assert adapter.embed([{"id": "a", "sequence": "AC"}, {"id": "b", "sequence": "ACD"}]) == {
        "a": [2.], "b": [4.],
    }


@pytest.mark.parametrize("model", ["esm2_t6_8M_UR50D", "esm2_t12_35M_UR50D"])
def test_checkpoint_verification_precedes_pickle_or_framework_loading(tmp_path, model):
    checkpoint = tmp_path / "fake.pt"
    checkpoint.write_bytes(b"not official checkpoint bytes")
    with pytest.raises(ValueError, match="pinned"):
        ESM2Embeddings(checkpoint, model_name=model)


def test_unknown_model_rejected_without_opening_checkpoint():
    with pytest.raises(ValueError, match="identity"):
        ESM2Embeddings("absent", model_name="unreviewed-model")
