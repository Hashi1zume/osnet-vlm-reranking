"""Build an identity-labeled embedding-pair manifest without touching test IDs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def iou(left: list[float], right: list[float]) -> float:
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rw, rh = right
    rx2, ry2 = rx1 + rw, ry1 + rh
    ix1, iy1, ix2, iy2 = max(lx1, rx1), max(ly1, ry1), min(lx2, rx2), min(ly2, ry2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_l = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    area_r = rw * rh
    return inter / max(area_l + area_r - inter, 1e-9)


def load_gt(path: Path) -> dict[int, list[tuple[int, list[float]]]]:
    result = defaultdict(list)
    for line in path.read_text().splitlines():
        fields = line.split(",")
        if len(fields) < 7 or int(fields[6]) != 1:
            continue
        result[int(fields[0])].append((int(fields[1]), [float(x) for x in fields[2:6]]))
    return result


def build(log: Path, gt_path: Path, output: Path, min_iou: float, excluded_ids: set[int]) -> int:
    gt = load_gt(gt_path)
    by_identity: dict[int, list[dict]] = defaultdict(list)
    for line in log.read_text().splitlines():
        record = json.loads(line)
        candidates = gt.get(record["frame"], [])
        matches = [(iou(record["bbox"], box), identity) for identity, box in candidates]
        if not matches or max(matches)[0] < min_iou:
            continue
        _, identity = max(matches)
        if identity not in excluded_ids:
            record["gt_identity"] = identity
            by_identity[identity].append(record)

    pairs = []
    identities = sorted(by_identity)
    for identity in identities:
        samples = by_identity[identity]
        for first, second in zip(samples, samples[1:]):
            pairs.append({"query_embedding": first["embedding"], "candidate_embedding": second["embedding"], "same_identity": True, "gt_identity": identity})
        for other in identities:
            if other == identity:
                continue
            candidate = by_identity[other][0]
            query_vector = np.asarray(samples[0]["embedding"])
            candidate_vector = np.asarray(candidate["embedding"])
            similarity = float(np.dot(query_vector, candidate_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(candidate_vector)))
            pairs.append({"query_embedding": samples[0]["embedding"], "candidate_embedding": candidate["embedding"], "same_identity": False, "gt_identity": identity, "negative_identity": other, "osnet_similarity": similarity})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(pair) for pair in pairs) + ("\n" if pairs else ""))
    return len(pairs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("gt", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--min-iou", type=float, default=0.5)
    parser.add_argument("--exclude-id", type=int, action="append", default=[])
    args = parser.parse_args()
    print(build(args.log, args.gt, args.output, args.min_iou, set(args.exclude_id)))


if __name__ == "__main__":
    main()
