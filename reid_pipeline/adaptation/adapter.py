import torch
from torch import nn


class LowRankAdapter(nn.Module):
    """Residual low-rank projection initialized as an identity map."""

    def __init__(self, dim: int = 512, rank: int = 16, alpha: float = 16.0):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        self.scale = alpha / rank
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return embedding + self.scale * self.up(self.down(embedding))
