from pathlib import Path

import pytest

from reid_pipeline.evaluation.reid_metrics import evaluate_retrieval
from reid_pipeline.data.mot import split_sequences


def test_evaluate_retrieval_computes_rank1_and_map():
    metrics = evaluate_retrieval(
        query_ids=["a", "b"],
        gallery_ids=["b", "a", "c"],
        scores=[[0.1, 0.9, 0.2], [0.8, 0.3, 0.1]],
    )
    assert metrics.rank1 == pytest.approx(1.0)
    assert metrics.mAP == pytest.approx(1.0)


def test_evaluate_retrieval_handles_empty_gallery():
    metrics = evaluate_retrieval(query_ids=["a"], gallery_ids=[], scores=[[]])
    assert metrics.rank1 == 0.0
    assert metrics.mAP == 0.0


def test_split_sequences_rejects_overlap():
    with pytest.raises(ValueError, match="overlap"):
        split_sequences(["s1", "s2"], ["s1"], ["s1"], [])


def test_split_sequences_keeps_all_requested_sequences():
    result = split_sequences(["s1", "s2", "s3"], ["s1"], ["s2"], ["s3"])
    assert result.adaptation == ("s1",)
    assert result.development == ("s2",)
    assert result.final == ("s3",)
