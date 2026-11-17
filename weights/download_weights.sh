#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEIGHTS_DIR="$ROOT_DIR/weights"
YOLO26_URL="https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt"
OSNET_MSMT17_URL="https://drive.google.com/uc?id=1sSwXSUlj4_tHZequ_iZ8w_Jh0VaRQMqF"
mkdir -p "$WEIGHTS_DIR"

if ! command -v gdown >/dev/null 2>&1; then
    echo "gdown is missing; first run: python -m pip install -r requirements.txt" >&2
    exit 1
fi

if [[ ! -s "$WEIGHTS_DIR/yolo26n.pt" ]]; then
    curl --fail --location --retry 3 --retry-all-errors \
        "$YOLO26_URL" --output "$WEIGHTS_DIR/yolo26n.pt"
fi
if [[ ! -s "$WEIGHTS_DIR/osnet_x0_25_msmt17.pth" ]]; then
    gdown "$OSNET_MSMT17_URL" --output "$WEIGHTS_DIR/osnet_x0_25_msmt17.pth" --continue --retries 3
fi

for WEIGHT_FILE in "$WEIGHTS_DIR/yolo26n.pt" "$WEIGHTS_DIR/osnet_x0_25_msmt17.pth"; do
    FILE_BYTES="$(wc -c < "$WEIGHT_FILE")"
    if (( FILE_BYTES < 500000 )); then
        echo "Downloaded file is unexpectedly small: $WEIGHT_FILE ($FILE_BYTES bytes)" >&2
        exit 1
    fi
    echo "Ready: $WEIGHT_FILE ($FILE_BYTES bytes)"
done
