import torch

from reid_pipeline.adaptation.adapter import LowRankAdapter


def test_low_rank_adapter_preserves_embedding_shape():
    adapter = LowRankAdapter(dim=8, rank=2, alpha=2)
    result = adapter(torch.ones(3, 8))
    assert result.shape == (3, 8)


def test_low_rank_adapter_is_identity_initialized():
    adapter = LowRankAdapter(dim=8, rank=2, alpha=2)
    inputs = torch.randn(3, 8)
    assert torch.allclose(adapter(inputs), inputs)


def test_low_rank_adapter_has_trainable_parameters():
    adapter = LowRankAdapter(dim=8, rank=2, alpha=2)
    assert sum(parameter.numel() for parameter in adapter.parameters()) == 32
