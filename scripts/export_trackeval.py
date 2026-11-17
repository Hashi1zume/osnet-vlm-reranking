"""Export the project's JSONL track audit into MOTChallenge text format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def export(source: Path, destination: Path) -> None:
    lines = []
    identity_map: dict[str, int] = {}
    next_identity = 1
    for raw in source.read_text().splitlines():
        record = json.loads(raw)
        identity = str(record.get("assigned_identity", record["track_id"]))
        if identity not in identity_map:
            identity_map[identity] = next_identity
            next_identity += 1
        x1, y1, x2, y2 = record["bbox"]
        lines.append(f"{record['frame']},{identity_map[identity]},{x1},{y1},{x2-x1},{y2-y1},1,-1,-1")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    export(args.source, args.destination)


if __name__ == "__main__":
    main()
