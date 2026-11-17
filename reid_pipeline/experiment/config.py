from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    video: Path
    output: Path
    mode: str = "osnet"
    frame_skip: int = 3

    def __post_init__(self):
        video = Path(self.video)
        if not video.exists():
            raise FileNotFoundError(video)
        if self.mode not in {"osnet", "rerank"}:
            raise ValueError(f"unsupported mode: {self.mode}")
        if self.frame_skip < 1:
            raise ValueError("frame_skip must be positive")
