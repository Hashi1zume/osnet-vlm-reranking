"""Mine high-similarity, GT-inconsistent candidates from a tracking audit."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx, by, bw, bh = b
    ax2b, ay2b = bx + bw, by + bh
    inter = max(0, min(ax2, ax2b) - max(ax1, bx)) * max(0, min(ay2, ay2b) - max(ay1, by))
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = bw * bh
    return inter / max(area_a + area_b - inter, 1e-9)


def load_gt(path):
    result = defaultdict(list)
    for line in path.read_text().splitlines():
        fields = line.split(",")
        if len(fields) >= 7 and int(fields[6]) == 1:
            result[int(fields[0])].append((int(fields[1]), [float(x) for x in fields[2:6]]))
    return result


def mine(log, gt_path, output, top_k):
    gt = load_gt(gt_path)
    records = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    gallery = {}
    dominant = defaultdict(Counter)
    labeled = []
    for record in records:
        matches = [(iou(record["bbox"], box), identity) for identity, box in gt[record["frame"]]]
        if not matches or max(matches)[0] < 0.5:
            continue
        _, gt_identity = max(matches)
        assigned = record["assigned_identity"]
        dominant[assigned][gt_identity] += 1
        gallery.setdefault(assigned, record)
        labeled.append((record, gt_identity))
    dominant_id = {key: counts.most_common(1)[0][0] for key, counts in dominant.items()}
    rows = []
    for record, gt_identity in labeled:
        query = np.asarray(record["embedding"], dtype=np.float32)
        query /= max(np.linalg.norm(query), 1e-9)
        candidates = []
        for assigned, candidate in gallery.items():
            if assigned == record["assigned_identity"]:
                continue
            vector = np.asarray(candidate["embedding"], dtype=np.float32)
            vector /= max(np.linalg.norm(vector), 1e-9)
            candidates.append((float(query @ vector), assigned, dominant_id[assigned]))
        for score, assigned, candidate_gt in sorted(candidates, reverse=True)[:top_k]:
            if candidate_gt != gt_identity:
                rows.append({"frame": record["frame"], "query_assigned": record["assigned_identity"], "candidate_assigned": assigned, "query_gt": gt_identity, "candidate_gt": candidate_gt, "osnet_similarity": score})
    rows.sort(key=lambda item: item["osnet_similarity"], reverse=True)
    output.write_text("\n".join(json.dumps(row) for row in rows) + ("\n" if rows else ""))
    return len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("gt", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    print(mine(args.log, args.gt, args.output, args.top_k))


if __name__ == "__main__":
    main()
