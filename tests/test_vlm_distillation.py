import torch

from reid_pipeline.adaptation.vlm_distillation import pair_relation_distillation_loss


def test_soft_vlm_relation_loss_is_finite_and_differentiable():
    query = torch.randn(4, 8, requires_grad=True)
    candidate = torch.randn(4, 8)
    teacher = torch.tensor([1.0, 0.0, 0.75, 0.25])
    loss = pair_relation_distillation_loss(query, candidate, teacher)
    loss.backward()
    assert torch.isfinite(loss)
    assert query.grad is not None
