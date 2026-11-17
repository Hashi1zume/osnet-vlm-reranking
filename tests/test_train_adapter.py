import json

import torch

from reid_pipeline.adaptation.train_adapter import AdapterTrainer, AdapterTrainingConfig, PairManifestDataset


def test_adapter_training_reduces_pair_loss(tmp_path):
    manifest = tmp_path / "pairs.jsonl"
    base = torch.zeros(8)
    base[0] = 1
    other = torch.zeros(8)
    other[1] = 1
    records = [
        {"query_embedding": base.tolist(), "candidate_embedding": base.tolist(), "same_identity": True},
        {"query_embedding": base.tolist(), "candidate_embedding": other.tolist(), "same_identity": False},
    ]
    manifest.write_text("\n".join(json.dumps(record) for record in records))
    dataset = PairManifestDataset(manifest, dim=8)
    config = AdapterTrainingConfig(dim=8, rank=2, alpha=2, epochs=3, batch_size=2)
    trainer = AdapterTrainer(config)
    history = trainer.fit(dataset)
    assert len(history) == 3
    assert history[-1] <= history[0]
