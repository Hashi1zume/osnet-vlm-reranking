#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY_DIR="$ROOT_DIR/third_party"
TORCHREID_DIR="$THIRD_PARTY_DIR/deep-person-reid"
TRACKEVAL_DIR="$THIRD_PARTY_DIR/TrackEval"
TORCHREID_COMMIT="f8cd150fdf77e8d9e1ed143b7f308c2c609ded50"
TRACKEVAL_COMMIT="12c8791b303e0a0b50f753af204249e622d0281a"

clone_at_commit() {
    local repository_url="$1"
    local destination="$2"
    local commit="$3"
    if [[ ! -d "$destination/.git" ]]; then
        mkdir -p "$(dirname "$destination")"
        git clone "$repository_url" "$destination"
    fi
    git -C "$destination" fetch origin "$commit"
    git -C "$destination" checkout --detach "$commit"
}

clone_at_commit "https://github.com/KaiyangZhou/deep-person-reid.git" "$TORCHREID_DIR" "$TORCHREID_COMMIT"
clone_at_commit "https://github.com/JonathonLuiten/TrackEval.git" "$TRACKEVAL_DIR" "$TRACKEVAL_COMMIT"

# This pinned TrackEval revision uses NumPy aliases removed in NumPy 2.
PATCH_FILE="$ROOT_DIR/patches/trackeval-numpy2.patch"
if ! git -C "$TRACKEVAL_DIR" apply --reverse --check "$PATCH_FILE" >/dev/null 2>&1; then
    git -C "$TRACKEVAL_DIR" apply "$PATCH_FILE"
fi

python -m pip install --no-deps --no-build-isolation -e "$TORCHREID_DIR"
python -m pip install --no-deps -e "$TRACKEVAL_DIR"
