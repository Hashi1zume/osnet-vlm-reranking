import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from reid_pipeline.retrieval.osnet_retriever import OSNetRetriever
from reid_pipeline.reranking.vlm_reranker import VLMReranker
from reid_pipeline.experiment.identity import IdentityAssigner
from reid_pipeline.temporal.gallery import TemporalIdentityGallery


class VideoReIDRunner:
    """Coordinates detector, encoder, and optional reranker without owning their setup."""

    def __init__(self, detector, encoder, config, reranker: VLMReranker | None = None, device="cuda"):
        self.detector = detector
        self.encoder = encoder.to(device).eval()
        self.config = config
        self.reranker = reranker
        self.device = device
        self.retriever = OSNetRetriever()
        self.identity_assigner = IdentityAssigner()
        self.temporal_gallery = TemporalIdentityGallery()
        self._track_to_global = {}
        self._next_global_id = 1
        self.transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])

    def run(self):
        cap = cv2.VideoCapture(str(self.config.video))
        if not cap.isOpened():
            raise RuntimeError(f"could not open video: {self.config.video}")
        self.config.output.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        frame_count = detection_count = vlm_calls = 0
        track_gallery = {}
        reranked_tracks = set()
        with self.config.output.open("w") as output:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_count % self.config.frame_skip:
                    frame_count += 1
                    continue
                result = self.detector.track(frame, persist=True, classes=[0], verbose=False)[0]
                if result.boxes is None:
                    frame_count += 1
                    continue
                boxes = result.boxes.xyxy.detach().cpu().numpy()
                ids = result.boxes.id.detach().cpu().numpy().astype(int) if result.boxes.is_track else range(len(boxes))
                crops, metadata = [], []
                height, width = frame.shape[:2]
                for track_id, box in zip(ids, boxes):
                    x1, y1, x2, y2 = map(int, box)
                    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
                    crop = frame[y1:y2, x1:x2]
                    if crop.size:
                        crops.append(self.transform(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))))
                        metadata.append((int(track_id), crop, (x1, y1, x2, y2)))
                if crops:
                    with torch.inference_mode():
                        embeddings = self.encoder(torch.stack(crops).to(self.device)).detach().cpu().numpy()
                    for embedding, (track_id, crop, bbox) in zip(embeddings, metadata):
                        rerank_result = None
                        if self.reranker is not None and track_id not in reranked_tracks and track_gallery:
                            gallery = {}
                            for gallery_id, item in track_gallery.items():
                                gallery[gallery_id] = item["embedding"]
                            candidates = self.retriever.retrieve(embedding.tolist(), gallery)
                            candidate_rows = []
                            query_bytes = cv2.imencode(".jpg", crop)[1].tobytes()
                            for candidate in candidates:
                                candidate_bytes = cv2.imencode(".jpg", track_gallery[candidate.identity]["image"])[1].tobytes()
                                candidate_rows.append((candidate.identity, candidate.osnet_score, 0.0, [candidate_bytes]))
                            rerank_result = self.reranker.rerank([query_bytes], candidate_rows)
                            reranked_tracks.add(track_id)
                            vlm_calls += 1
                        if track_id not in self._track_to_global:
                            matched_identity = None
                            if rerank_result and self.reranker is not None:
                                for candidate_id, osnet_score, _ in rerank_result:
                                    judgment = self.reranker.last_judgments.get(candidate_id)
                                    # Fusion ranks candidates; the OSNet floor stays independent because
                                    # the current video path has no attribute score to contribute.
                                    if (
                                        judgment is not None
                                        and judgment.same_person
                                        and osnet_score >= self.identity_assigner.minimum_score
                                    ):
                                        matched_identity = self._track_to_global.get(
                                            candidate_id, f"global-{candidate_id}"
                                        )
                                        break
                            # VLM mode rejects unapproved merges; OSNet mode retains the temporal baseline.
                            if matched_identity is None and self.reranker is None:
                                matched_identity, _ = self.temporal_gallery.match(embedding, frame_count + 1)
                            if matched_identity is None:
                                matched_identity = f"global-{self._next_global_id}"
                                self._next_global_id += 1
                            self._track_to_global[track_id] = matched_identity
                        assigned_identity = self._track_to_global[track_id]
                        self.temporal_gallery.update(assigned_identity, embedding, frame_count + 1)
                        track_gallery.setdefault(track_id, {"embedding": embedding.tolist(), "image": crop})
                        candidate_judgments = []
                        if rerank_result and self.reranker is not None:
                            for candidate_id, _, _ in rerank_result:
                                judgment = self.reranker.last_judgments.get(candidate_id)
                                candidate_judgments.append({
                                    "track_id": candidate_id,
                                    "same_person": judgment.same_person if judgment else None,
                                    "confidence": judgment.confidence if judgment else None,
                                })
                        # JSONL is an audit trail so each feature can be traced back to its source frame.
                        output.write(json.dumps({
                            "frame": frame_count + 1,
                            "track_id": track_id,
                            "bbox": bbox,
                            "embedding": embedding.tolist(),
                            "reranked": rerank_result,
                            "candidate_judgments": candidate_judgments,
                            "assigned_identity": assigned_identity,
                        }) + "\n")
                        detection_count += 1
                frame_count += 1
            cap.release()
        elapsed = time.perf_counter() - started
        summary = {"frames": frame_count, "detections": detection_count, "tracks": len(track_gallery), "vlm_calls": vlm_calls, "fps": frame_count / elapsed if elapsed else 0.0}
        with self.config.output.with_suffix(".summary.json").open("w") as handle:
            json.dump(summary, handle, indent=2)
        return summary
