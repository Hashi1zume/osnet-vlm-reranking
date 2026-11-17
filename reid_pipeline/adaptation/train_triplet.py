"""Batch-hard triplet training for identity-labeled OSNet embeddings."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

from .adapter import LowRankAdapter


@dataclass(frozen=True)
class TripletConfig:
    dim: int = 512
    rank: int = 16
    alpha: float = 16.0
    epochs: int = 20
    learning_rate: float = 1e-4
    margin: float = 0.2
    identity_regularization: float = 10.0
    seed: int = 7


def load_manifest(path: Path, dim: int) -> tuple[Tensor, Tensor]:
    vectors, labels = [], []
    for line in path.read_text().splitlines():
        record = json.loads(line)
        if "embedding" not in record or "gt_identity" not in record:
            continue
        vector = torch.tensor(record["embedding"], dtype=torch.float32)
        if vector.numel() == dim:
            vectors.append(F.normalize(vector, dim=0))
            labels.append(int(record["gt_identity"]))
    if len(set(labels)) < 2:
        raise ValueError("triplet training requires at least two identities")
    return torch.stack(vectors), torch.tensor(labels)


class BatchHardTrainer:
    """Learns a conservative residual correction while OSNet remains frozen."""

    def __init__(self, config: TripletConfig, device: str = "cpu"):
        self.config = config
        self.device = torch.device(device)
        torch.manual_seed(config.seed)
        self.adapter = LowRankAdapter(config.dim, config.rank, config.alpha).to(self.device)

    def fit(self, embeddings: Tensor, labels: Tensor) -> list[float]:
        embeddings, labels = embeddings.to(self.device), labels.to(self.device)
        optimizer = torch.optim.AdamW(self.adapter.parameters(), lr=self.config.learning_rate)
        history = []
        for _ in range(self.config.epochs):
            adapted = F.normalize(self.adapter(embeddings), dim=1)
            distances = 1.0 - adapted @ adapted.T
            same = labels[:, None] == labels[None, :]
            positive_distances = distances.masked_fill(~same, -1.0)
            negative_distances = distances.masked_fill(same, 2.0)
            hard_positive = positive_distances.max(dim=1).values
            hard_negative = negative_distances.min(dim=1).values
            valid = (positive_distances >= 0).sum(dim=1) > 1
            triplet = F.relu(hard_positive - hard_negative + self.config.margin)[valid].mean()
            identity_loss = F.mse_loss(self.adapter(embeddings), embeddings)
            loss = triplet + self.config.identity_regularization * identity_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            history.append(float(loss.detach().cpu()))
        return history

    def save(self, path: Path, history: list[float]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path_pt = {"state_dict": self.adapter.state_dict(), "config": asdict(self.config), "loss": history}
        torch.save(path_pt, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    config = TripletConfig()
    trainer = BatchHardTrainer(config, args.device)
    embeddings, labels = load_manifest(args.manifest, config.dim)
    history = trainer.fit(embeddings, labels)
    trainer.save(args.output, history)
    print(json.dumps({"output": str(args.output), "final_loss": history[-1], "samples": len(labels)}))


if __name__ == "__main__":
    main()
