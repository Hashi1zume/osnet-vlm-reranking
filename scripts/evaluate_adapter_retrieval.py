"""Compare frozen OSNet and adapted embeddings on a labeled MOT log."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from reid_pipeline.adaptation.adapter import LowRankAdapter
from reid_pipeline.evaluation.reid_metrics import evaluate_retrieval


def load_records(path: Path):
    groups = {}
    for line in path.read_text().splitlines():
        record = json.loads(line)
        groups.setdefault(record["assigned_identity"], []).append(record)
    return groups


def evaluate(groups, adapter=None):
    gallery = {}
    queries = []
    for identity, records in sorted(groups.items()):
        if len(records) < 2:
            continue
        gallery[identity] = records[0]["embedding"]
        queries.append((identity, records[1:]))
    if adapter is not None:
        gallery = {key: adapter(F.normalize(torch.tensor(value), dim=0)).detach().tolist() for key, value in gallery.items()}
    scores = []
    query_ids = []
    for identity, records in queries:
        for record in records:
            vector = torch.tensor(record["embedding"])
            if adapter is not None:
                vector = adapter(F.normalize(vector, dim=0)).detach()
            scores.append([float(torch.dot(F.normalize(vector, dim=0), F.normalize(torch.tensor(value), dim=0))) for value in gallery.values()])
            query_ids.append(identity)
    return evaluate_retrieval(query_ids, list(gallery), scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    groups = load_records(args.log)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    adapter = LowRankAdapter(config["dim"], config["rank"], config["alpha"])
    adapter.load_state_dict(checkpoint["state_dict"])
    adapter.eval()
    baseline = evaluate(groups)
    adapted = evaluate(groups, adapter)
    print(json.dumps({"osnet": baseline.__dict__, "adapter": adapted.__dict__}, indent=2))


if __name__ == "__main__":
    main()
