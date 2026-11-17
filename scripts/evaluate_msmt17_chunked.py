"""Memory-bounded MSMT17 evaluation for the frozen OSNet baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

REID_ROOT = Path(__file__).resolve().parents[1] / "vendor" / "deep-person-reid"
sys.path.insert(0, str(REID_ROOT / "scripts"))
import torchreid  # noqa: E402
from default_config import get_default_config, imagedata_kwargs  # noqa: E402


def extract(model, loader, use_gpu: bool):
    features, pids, cams = [], [], []
    with torch.inference_mode():
        for batch in loader:
            imgs, batch_pids, batch_cams = batch["img"], batch["pid"], batch["camid"]
            if use_gpu:
                imgs = imgs.cuda(non_blocking=True)
            output = model(imgs)
            if isinstance(output, tuple):
                output = output[0]
            features.append(output.detach().cpu())
            pids.extend(batch_pids.tolist())
            cams.extend(batch_cams.tolist())
    return torch.cat(features), np.asarray(pids), np.asarray(cams)


def evaluate_chunked(query, q_pids, q_cams, gallery, g_pids, g_cams, chunk_size: int):
    rank1, average_precisions, valid = 0, [], 0
    gallery = F.normalize(gallery, dim=1)
    query = F.normalize(query, dim=1)
    for start in range(0, len(query), chunk_size):
        q = query[start:start + chunk_size]
        distances = 1.0 - q @ gallery.T
        for row, q_pid, q_cam in zip(distances, q_pids[start:start + chunk_size], q_cams[start:start + chunk_size]):
            order = torch.argsort(row).numpy()
            keep = ~((g_pids[order] == q_pid) & (g_cams[order] == q_cam))
            matches = (g_pids[order][keep] == q_pid)
            if not matches.any():
                continue
            valid += 1
            rank1 += int(matches[0])
            cumulative = np.cumsum(matches)
            positions = np.arange(1, len(matches) + 1)
            average_precisions.append(float((cumulative / positions * matches).sum() / matches.sum()))
    return {"rank1": rank1 / valid, "mAP": float(np.mean(average_precisions)), "valid_queries": valid}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=128)
    args = parser.parse_args()
    cfg = get_default_config()
    cfg.data.root = str(args.root)
    cfg.data.sources = ["msmt17"]
    cfg.data.targets = ["msmt17"]
    cfg.data.type = "image"
    cfg.data.height, cfg.data.width = 256, 128
    cfg.test.batch_size = 256
    cfg.use_gpu = torch.cuda.is_available()
    manager = torchreid.data.ImageDataManager(**imagedata_kwargs(cfg))
    model = torchreid.models.build_model("osnet_x0_25", manager.num_train_pids, loss="softmax", pretrained=True, use_gpu=cfg.use_gpu)
    model.eval()
    if cfg.use_gpu:
        model = model.cuda()
    loaders = manager.test_loader["msmt17"]
    query = extract(model, loaders["query"], cfg.use_gpu)
    gallery = extract(model, loaders["gallery"], cfg.use_gpu)
    result = evaluate_chunked(*query, *gallery, args.chunk_size)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
