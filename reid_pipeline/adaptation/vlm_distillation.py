"""Losses for transferring VLM pair relations into a frozen-encoder student."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def pair_relation_distillation_loss(
    query: Tensor,
    candidate: Tensor,
    teacher_same_probability: Tensor,
    margin: float = 0.3,
) -> Tensor:
    """Match student cosine similarity to a soft VLM relation score.

    The teacher probability is intentionally kept soft: uncertain VLM outputs
    should move the OSNet representation less than repeated high-confidence
    judgments. This avoids converting a noisy VLM response into a hard label.
    """
    query = F.normalize(query, dim=1)
    candidate = F.normalize(candidate, dim=1)
    student_similarity = ((query * candidate).sum(dim=1) + 1.0) / 2.0
    relation_loss = F.smooth_l1_loss(student_similarity, teacher_same_probability)
    distance = 1.0 - (query * candidate).sum(dim=1)
    hard_negative_loss = F.relu(margin - distance) * (1.0 - teacher_same_probability)
    return relation_loss + hard_negative_loss.mean()
