import pytest

from reid_pipeline.adaptation.hard_cases import collect_hard_cases


def test_collects_high_confidence_conflict_as_hard_negative():
    cases = collect_hard_cases(
        [{"sequence": "adapt", "query_id": 1, "candidate_id": 2, "osnet_rank": 1, "same_identity": False}],
        [{"same_person": False, "confidence": 0.9}],
        final_sequences={"final"},
    )
    assert len(cases) == 1
    assert cases[0]["kind"] == "hard_negative"


def test_rejects_final_sequence_cases():
    with pytest.raises(ValueError, match="final"):
        collect_hard_cases(
            [{"sequence": "final", "query_id": 1, "candidate_id": 2, "osnet_rank": 1, "same_identity": False}],
            [{"same_person": False, "confidence": 0.9}],
            final_sequences={"final"},
        )
