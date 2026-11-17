#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIDEO_PATH="${1:?Usage: validate_mot20.sh VIDEO_PATH TRACK_EVAL_GT_ROOT [SEQUENCE]}"
GT_ROOT="${2:?Usage: validate_mot20.sh VIDEO_PATH TRACK_EVAL_GT_ROOT [SEQUENCE]}"
SEQUENCE_NAME="${3:-MOT20-01}"
DETECTOR_WEIGHTS="${DETECTOR_WEIGHTS:-$ROOT_DIR/weights/yolo26n.pt}"
OSNET_WEIGHTS="${OSNET_WEIGHTS:-$ROOT_DIR/weights/osnet_x0_25_msmt17.pth}"
VLM_MODEL="${VLM_MODEL:-qwen2.5vl:7b}"
TRACKER_ROOT="$ROOT_DIR/runs/trackeval/trackers"
TRACKER_SET="$TRACKER_ROOT/MOT20-train"
TRACKEVAL_SCRIPT="$ROOT_DIR/third_party/TrackEval/scripts/run_mot_challenge.py"

if [[ ! -f "$VIDEO_PATH" ]]; then
    echo "Video not found: $VIDEO_PATH" >&2
    exit 1
fi
if [[ ! -f "$GT_ROOT/MOT20-train/$SEQUENCE_NAME/seqinfo.ini" ]]; then
    echo "Expected TrackEval GT at $GT_ROOT/MOT20-train/$SEQUENCE_NAME/seqinfo.ini" >&2
    exit 1
fi
if [[ ! -f "$DETECTOR_WEIGHTS" || ! -f "$OSNET_WEIGHTS" ]]; then
    echo "Run weights/download_weights.sh or set DETECTOR_WEIGHTS and OSNET_WEIGHTS." >&2
    exit 1
fi

mkdir -p "$TRACKER_SET"
for MODE in osnet rerank; do
    if [[ "$MODE" == "osnet" ]]; then
        TRACKER_NAME="temporal_osnet"
    else
        TRACKER_NAME="vlm_gate"
    fi
    OUTPUT_JSONL="$ROOT_DIR/runs/$TRACKER_NAME.jsonl"
    RAW_TRACKER="$TRACKER_SET/$TRACKER_NAME/data/$SEQUENCE_NAME.txt"
    CLEAN_TRACKER="$TRACKER_SET/$TRACKER_NAME-clean/data/$SEQUENCE_NAME.txt"

    python "$ROOT_DIR/scripts/run_video_reid.py" \
        --video "$VIDEO_PATH" \
        --output "$OUTPUT_JSONL" \
        --detector "$DETECTOR_WEIGHTS" \
        --mode "$MODE" \
        --frame-skip 1 \
        --vlm-model "$VLM_MODEL" \
        --osnet-weights "$OSNET_WEIGHTS"
    python "$ROOT_DIR/scripts/export_trackeval.py" "$OUTPUT_JSONL" "$RAW_TRACKER"
    python "$ROOT_DIR/scripts/sanitize_trackeval_ids.py" "$RAW_TRACKER" "$CLEAN_TRACKER"
done

python "$TRACKEVAL_SCRIPT" \
    --BENCHMARK MOT20 \
    --SPLIT_TO_EVAL train \
    --TRACKERS_TO_EVAL temporal_osnet-clean vlm_gate-clean \
    --METRICS HOTA CLEAR Identity \
    --USE_PARALLEL False \
    --NUM_PARALLEL_CORES 1 \
    --GT_FOLDER "$GT_ROOT" \
    --TRACKERS_FOLDER "$TRACKER_ROOT" \
    --SEQ_INFO "$SEQUENCE_NAME"
