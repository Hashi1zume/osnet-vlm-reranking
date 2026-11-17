import pytest

from reid_pipeline.experiment.config import ExperimentConfig


def test_config_requires_existing_video(tmp_path):
    with pytest.raises(FileNotFoundError):
        ExperimentConfig(video=tmp_path / "missing.mp4", output=tmp_path / "out.jsonl")


def test_config_accepts_video_and_normalizes_mode(tmp_path):
    video = tmp_path / "input.mp4"
    video.touch()
    config = ExperimentConfig(video=video, output=tmp_path / "out.jsonl", mode="rerank")
    assert config.mode == "rerank"
