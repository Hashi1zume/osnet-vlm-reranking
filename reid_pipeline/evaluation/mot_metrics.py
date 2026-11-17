from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class MOTMetrics:
    true_positive: int
    false_positive: int
    false_negative: int
    precision: float
    recall: float
    track_purity: float
    id_switches: int


def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union else 0.0


def _prediction_bbox(prediction):
    x1, y1, x2, y2 = prediction["bbox"]
    return (x1, y1, x2 - x1, y2 - y1)


def evaluate_detections(ground_truth, predictions, iou_threshold=0.5):
    by_frame = defaultdict(list)
    for prediction in predictions:
        by_frame[prediction["frame"]].append(prediction)
    matches = []
    tp = fp = fn = 0
    for frame, truth in ground_truth.items():
        candidates = by_frame.get(frame, [])
        pairs = sorted(
            (_iou(_prediction_bbox(pred), bbox), pred, identity)
            for pred in candidates
            for identity, bbox in truth
            if _iou(_prediction_bbox(pred), bbox) >= iou_threshold
        )[::-1]
        used_predictions, used_truth = set(), set()
        for score, prediction, identity in pairs:
            marker = id(prediction)
            if marker in used_predictions or identity in used_truth:
                continue
            used_predictions.add(marker)
            used_truth.add(identity)
            matches.append((prediction.get("assigned_identity", prediction["track_id"]), identity))
        tp += len(used_predictions)
        fp += len(candidates) - len(used_predictions)
        fn += len(truth) - len(used_truth)
    for frame, candidates in by_frame.items():
        if frame not in ground_truth:
            fp += len(candidates)
    track_labels = defaultdict(list)
    for track_id, identity in matches:
        track_labels[track_id].append(identity)
    consistent = sum(max(Counter(labels).values()) for labels in track_labels.values())
    total = sum(len(labels) for labels in track_labels.values())
    switches = sum(max(0, len(labels) - 1) for labels in track_labels.values() if len(set(labels)) > 1)
    return MOTMetrics(tp, fp, fn, tp / (tp + fp) if tp + fp else 0.0, tp / (tp + fn) if tp + fn else 0.0, consistent / total if total else 0.0, switches)


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
