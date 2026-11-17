"""Label mined hard-negative pairs with the local Qwen2.5-VL teacher."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from reid_pipeline.reranking.ollama_client import OllamaVLMClient
from reid_pipeline.reranking.vlm_reranker import parse_vlm_json


def frame_images(video: Path, frame_numbers: set[int], records: dict[tuple[int, str], dict]) -> dict[tuple[int, str], bytes]:
    capture = cv2.VideoCapture(str(video))
    result = {}
    current = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        current += 1
        if current not in frame_numbers:
            continue
        for key, record in records.items():
            if key[0] != current:
                continue
            x1, y1, x2, y2 = map(int, record["bbox"])
            crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            if crop.size:
                result[key] = cv2.imencode(".jpg", crop)[1].tobytes()
    capture.release()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pairs", type=Path)
    parser.add_argument("audit", type=Path)
    parser.add_argument("video", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--model", default="qwen2.5vl:7b")
    args = parser.parse_args()
    candidates = [json.loads(line) for line in args.pairs.read_text().splitlines() if line.strip()][:args.limit]
    audit = {(r["frame"], r["assigned_identity"]): r for r in (json.loads(line) for line in args.audit.read_text().splitlines() if line.strip())}
    keys = {(item["frame"], item["query_assigned"]) for item in candidates} | {(item["frame"], item["candidate_assigned"]) for item in candidates}
    images = frame_images(args.video, {key[0] for key in keys}, audit)
    client = OllamaVLMClient(args.model)
    rows = []
    for item in candidates:
        query = images.get((item["frame"], item["query_assigned"]))
        candidate = images.get((item["frame"], item["candidate_assigned"]))
        if not query or not candidate:
            continue
        prompt = "Compare these two person crops. Return JSON with same_person, confidence, matched_attributes, conflicting_attributes, and unknown_attributes."
        raw = client.generate([query, candidate], prompt)
        judgment = parse_vlm_json(raw, minimum_confidence=0.0)
        if judgment:
            rows.append({**item, "vlm": {"same_person": judgment.same_person, "confidence": judgment.confidence, "matched_attributes": judgment.matched_attributes, "conflicting_attributes": judgment.conflicting_attributes, "unknown_attributes": judgment.unknown_attributes}})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(row) for row in rows) + ("\n" if rows else ""))
    print(json.dumps({"requested": len(candidates), "labeled": len(rows)}))


if __name__ == "__main__":
    main()
