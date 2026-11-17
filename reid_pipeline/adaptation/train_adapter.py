"""Train the frozen-OSNet embedding adapter from pairwise supervision."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .adapter import LowRankAdapter


@dataclass(frozen=True)
class AdapterTrainingConfig:
    """Training settings kept serializable for exact experiment replay."""

    dim: int = 512
    rank: int = 16
    alpha: float = 16.0
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-4
    margin: float = 0.3
    identity_regularization: float = 0.1
    seed: int = 7


class PairManifestDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """Loads normalized OSNet embedding pairs and a same-identity label."""

    def __init__(self, manifest: Path, dim: int):
        self.records = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        if not self.records:
            raise ValueError(f"manifest is empty: {manifest}")
        self.dim = dim

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        record = self.records[index]
        query = torch.tensor(record["query_embedding"], dtype=torch.float32)
        candidate = torch.tensor(record["candidate_embedding"], dtype=torch.float32)
        if query.numel() != self.dim or candidate.numel() != self.dim:
            raise ValueError("embedding dimension does not match training configuration")
        label = torch.tensor(float(record["same_identity"]), dtype=torch.float32)
        return F.normalize(query, dim=0), F.normalize(candidate, dim=0), label


class AdapterTrainer:
    """Optimizes only the residual low-rank adapter, leaving OSNet frozen."""

    def __init__(self, config: AdapterTrainingConfig, device: str = "cpu"):
        self.config = config
        self.device = torch.device(device)
        torch.manual_seed(config.seed)
        self.adapter = LowRankAdapter(config.dim, config.rank, config.alpha).to(self.device)

    def _loss(self, query: Tensor, candidate: Tensor, label: Tensor) -> Tensor:
        adapted_query = F.normalize(self.adapter(query), dim=1)
        adapted_candidate = F.normalize(self.adapter(candidate), dim=1)
        distance = 1.0 - (adapted_query * adapted_candidate).sum(dim=1)
        positive = label * distance.square()
        negative = (1.0 - label) * F.relu(self.config.margin - distance).square()
        pair_loss = (positive + negative).mean()
        # Preserve the pretrained embedding geometry while learning only the
        # correction needed for hard cases; this prevents low-rank collapse.
        adapted = self.adapter(torch.cat((query, candidate), dim=0))
        original = torch.cat((query, candidate), dim=0)
        identity_loss = F.mse_loss(adapted, original)
        return pair_loss + self.config.identity_regularization * identity_loss

    def fit(self, dataset: Dataset[tuple[Tensor, Tensor, Tensor]]) -> list[float]:
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(self.adapter.parameters(), lr=self.config.learning_rate)
        history: list[float] = []
        self.adapter.train()
        for _ in range(self.config.epochs):
            losses = []
            for query, candidate, label in loader:
                loss = self._loss(query.to(self.device), candidate.to(self.device), label.to(self.device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            history.append(sum(losses) / len(losses))
        return history

    def save(self, output: Path, history: list[float]) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.adapter.state_dict(), "config": asdict(self.config), "loss": history}, output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the OSNet residual low-rank adapter")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    config = AdapterTrainingConfig()
    trainer = AdapterTrainer(config, args.device)
    history = trainer.fit(PairManifestDataset(args.manifest, config.dim))
    trainer.save(args.output, history)
    print(json.dumps({"output": str(args.output), "final_loss": history[-1], "epochs": config.epochs}))


if __name__ == "__main__":
    main()
