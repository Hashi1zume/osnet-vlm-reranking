"""Make tracker exports valid for MOT/TrackEval without using ground truth.

TrackEval requires at most one detection per identity in a frame.  When the
experimental assigner emits duplicate IDs, later occurrences are split into
new IDs so the standard metrics expose the resulting fragmentation.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path


def sanitize(source: Path, destination: Path) -> None:
    seen: dict[int, set[int]] = defaultdict(set)
    next_fragment = 1_000_000
    output: list[str] = []
    for line in source.read_text().splitlines():
        if not line.strip():
            continue
        fields = line.split(",")
        frame = int(fields[0])
        identity = int(fields[1])
        if identity in seen[frame]:
            identity = next_fragment
            next_fragment += 1
        seen[frame].add(identity)
        fields[1] = str(identity)
        output.append(",".join(fields))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(output) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    sanitize(args.source, args.destination)


if __name__ == "__main__":
    main()
