"""Convert VLM hard-negative decisions into validated OSNet pair records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("vlm_labels", type=Path)
    parser.add_argument("audit", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--minimum-confidence", type=float, default=0.8)
    args = parser.parse_args()
    audit = {(r["frame"], r["assigned_identity"]): r for r in (json.loads(line) for line in args.audit.read_text().splitlines() if line.strip())}
    output = []
    for line in args.vlm_labels.read_text().splitlines():
        record = json.loads(line)
        judgment = record["vlm"]
        # Keep only VLM-confirmed negatives that also satisfy the mined GT constraint.
        if judgment["same_person"] or judgment["confidence"] < args.minimum_confidence:
            continue
        query = audit.get((record["frame"], record["query_assigned"]))
        candidate = audit.get((record["frame"], record["candidate_assigned"]))
        if query is None or candidate is None:
            continue
        output.append({"sequence": "MOT20-01", "query_id": record["query_gt"], "candidate_id": record["candidate_gt"], "query_embedding": query["embedding"], "candidate_embedding": candidate["embedding"], "same_identity": False, "vlm_confidence": judgment["confidence"]})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(row) for row in output) + ("\n" if output else ""))
    print(len(output))


if __name__ == "__main__":
    main()
