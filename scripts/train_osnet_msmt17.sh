#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TORCHREID_DIR="$ROOT_DIR/third_party/deep-person-reid"

if [[ ! -f "$ROOT_DIR/datasets/msmt17/MSMT17_V1/list_train.txt" ]]; then
    echo "MSMT17 V1 was not found under datasets/msmt17/MSMT17_V1" >&2
    exit 1
fi

cd "$ROOT_DIR"
python "$TORCHREID_DIR/scripts/main.py" \
    --config-file "$ROOT_DIR/configs/osnet_msmt17_train.yaml" \
    --root "$ROOT_DIR/datasets"
