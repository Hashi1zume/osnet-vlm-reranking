#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from ultralytics import YOLO
import torchreid
from torchreid.utils import load_pretrained_weights

from reid_pipeline.experiment.config import ExperimentConfig
from reid_pipeline.experiment.video_runner import VideoReIDRunner
from reid_pipeline.reranking.ollama_client import OllamaVLMClient
from reid_pipeline.reranking.vlm_reranker import VLMReranker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--detector", default="yolo26n.pt")
    parser.add_argument("--frame-skip", type=int, default=3)
    parser.add_argument("--mode", choices=["osnet", "rerank"], default="osnet")
    parser.add_argument("--vlm-model", default="qwen2.5vl:7b")
    parser.add_argument("--osnet-weights", type=Path, default=None)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = ExperimentConfig(args.video, args.output, mode=args.mode, frame_skip=args.frame_skip)
    detector = YOLO(args.detector)
    encoder = torchreid.models.build_model("osnet_x0_25", num_classes=1000, pretrained=True)
    if args.osnet_weights:
        # Training checkpoints contain optimizer metadata that PyTorch 2.6+ rejects
        # under weights-only loading; this file is locally generated and trusted.
        checkpoint = torch.load(args.osnet_weights, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
        state_dict = {key: value for key, value in state_dict.items() if not key.startswith("classifier.")}
        encoder.load_state_dict(state_dict, strict=False)
    reranker = VLMReranker(OllamaVLMClient(args.vlm_model)) if args.mode == "rerank" else None
    print(VideoReIDRunner(detector, encoder, config, reranker=reranker, device=device).run())


if __name__ == "__main__":
    main()
