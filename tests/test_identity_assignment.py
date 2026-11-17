from reid_pipeline.experiment.identity import IdentityAssigner


def test_assigner_reuses_candidate_above_threshold():
    assigner = IdentityAssigner(minimum_score=0.75)
    assert assigner.assign(10, [("old-1", 0.8, 0.81)]) == "old-1"


def test_assigner_keeps_new_track_below_threshold():
    assigner = IdentityAssigner(minimum_score=0.75)
    assert assigner.assign(10, [("old-1", 0.8, 0.74)]) == "track-10"


def test_assigner_keeps_assignment_stable():
    assigner = IdentityAssigner(minimum_score=0.75)
    assigner.assign(10, [("old-1", 0.8, 0.81)])
    assert assigner.assign(10, []) == "old-1"
