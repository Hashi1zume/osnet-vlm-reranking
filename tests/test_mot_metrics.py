from reid_pipeline.evaluation.mot_metrics import evaluate_detections


def test_evaluate_detections_matches_boxes_and_reports_track_purity():
    gt = {1: [(1, (0, 0, 10, 10))]}
    predictions = [{"frame": 1, "track_id": 7, "bbox": [0, 0, 10, 10]}]
    result = evaluate_detections(gt, predictions)
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.track_purity == 1.0


def test_evaluate_detections_counts_missed_ground_truth():
    result = evaluate_detections({1: [(1, (0, 0, 10, 10))]}, [])
    assert result.precision == 0.0
    assert result.recall == 0.0
