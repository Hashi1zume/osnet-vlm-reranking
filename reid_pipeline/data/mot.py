from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Split:
    adaptation: tuple[str, ...]
    development: tuple[str, ...]
    final: tuple[str, ...]


def split_sequences(all_sequences, adaptation, development, final) -> Split:
    groups = {
        "adaptation": tuple(adaptation),
        "development": tuple(development),
        "final": tuple(final),
    }
    seen: set[str] = set()
    for name, values in groups.items():
        unknown = set(values) - set(all_sequences)
        if unknown:
            raise ValueError(f"unknown sequences in {name}: {sorted(unknown)}")
        overlap = seen.intersection(values)
        if overlap:
            raise ValueError(f"overlap between sequence splits: {sorted(overlap)}")
        seen.update(values)
    return Split(**groups)


def load_mot_sequence(root: Path) -> list[dict]:
    """Load pedestrian GT rows from a MOT sequence directory."""
    gt_path = root / "gt.txt"
    rows = []
    if not gt_path.exists():
        return rows
    for line in gt_path.read_text().splitlines():
        fields = line.split(",")
        if len(fields) < 7:
            continue
        frame, identity = int(fields[0]), int(fields[1])
        x, y, width, height = map(float, fields[2:6])
        confidence = float(fields[6])
        category = int(fields[7]) if len(fields) > 7 else 1
        visibility = float(fields[8]) if len(fields) > 8 else 1.0
        if category == 1 and confidence > 0 and visibility > 0:
            rows.append({"frame": frame, "identity": identity, "bbox": (x, y, width, height)})
    return rows
