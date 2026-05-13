#!/bin/bash
# Download pretrained weights for OmniGen2 / SGT-Gen2 to a local directory.
# After running, point options/*.yml at the local paths:
#   pretrained_vae_model_name_or_path:          pretrained_models/FLUX.1-dev
#   pretrained_text_encoder_model_name_or_path: pretrained_models/Qwen2.5-VL-3B-Instruct
#   pretrained_model_path:                      pretrained_models/OmniGen2
# Pass --skip-flux / --skip-qwen / --skip-omnigen to skip individual repos.

set -euo pipefail

SHELL_FOLDER=$(cd "$(dirname "$0")"; pwd)
# Project root is scripts/../ -> OmniGen2/
PROJECT_ROOT=$(cd "$SHELL_FOLDER/.."; pwd)
cd "$PROJECT_ROOT"

# -------------------- Config (can be overridden by env / flags) --------------
TARGET_DIR=${TARGET_DIR:-"$PROJECT_ROOT/pretrained_models"}
# Use a writable HF cache to stage downloads. We deliberately override HF_HOME
# because the default one in this container is read-only.
HF_LOCAL_CACHE=${HF_LOCAL_CACHE:-"$TARGET_DIR/.hf_cache"}

SKIP_OMNIGEN=false
SKIP_FLUX=false
SKIP_QWEN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --target-dir=*)      TARGET_DIR="${1#*=}"; shift ;;
        --cache-dir=*)       HF_LOCAL_CACHE="${1#*=}"; shift ;;
        --skip-omnigen)      SKIP_OMNIGEN=true; shift ;;
        --skip-flux)         SKIP_FLUX=true; shift ;;
        --skip-qwen)         SKIP_QWEN=true; shift ;;
        -h|--help)
            sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

mkdir -p "$TARGET_DIR" "$HF_LOCAL_CACHE"

# -------------------- HF runtime fixes --------------------------------------
# Redirect HF cache to a writable local directory.
export HF_HOME="$HF_LOCAL_CACHE"
export HF_HUB_CACHE="$HF_LOCAL_CACHE/hub"
export HUGGINGFACE_HUB_CACHE="$HF_LOCAL_CACHE/hub"
# Disable the Xet backend to avoid hf_xet log-dir panics in restricted envs.
export HF_HUB_DISABLE_XET=1
# Friendlier download behaviour on flaky networks.
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-0}
# Optional: export HF_TOKEN="<your token>" before running this script if the
# model requires authentication (e.g. gated repos).

echo "=========================================================="
echo "  Downloading OmniGen2 pretrained weights"
echo "  TARGET_DIR     = $TARGET_DIR"
echo "  HF cache       = $HF_LOCAL_CACHE"
echo "  HF_HUB_DISABLE_XET = $HF_HUB_DISABLE_XET"
echo "=========================================================="

# -------------------- Helpers -----------------------------------------------
have_huggingface_cli() {
    command -v huggingface-cli >/dev/null 2>&1
}

# Snapshot-download a whole repo to a local directory using the CLI if
# available (best behaviour: resume + parallel), otherwise fall back to the
# Python API.
download_repo() {
    local repo_id="$1"
    local local_dir="$2"
    shift 2
    local extra_args=("$@")

    if [ -d "$local_dir" ] && [ -n "$(ls -A "$local_dir" 2>/dev/null || true)" ]; then
        echo "[skip] $repo_id already present at $local_dir"
        return 0
    fi

    mkdir -p "$local_dir"

    if have_huggingface_cli; then
        echo "[hf-cli] $repo_id -> $local_dir"
        huggingface-cli download "$repo_id" \
            --local-dir "$local_dir" \
            --local-dir-use-symlinks False \
            --resume-download \
            "${extra_args[@]}"
    else
        echo "[py-api] $repo_id -> $local_dir"
        python - <<PY
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="$repo_id",
    local_dir="$local_dir",
    local_dir_use_symlinks=False,
    resume_download=True,
    allow_patterns=${EXTRA_ALLOW_PATTERNS:-None},
)
PY
    fi
}

# -------------------- 1. OmniGen2 base checkpoint ---------------------------
OMNIGEN_DIR="$TARGET_DIR/OmniGen2"
if [ "$SKIP_OMNIGEN" = "true" ]; then
    echo "[skip] OmniGen2 download skipped by flag."
else
    download_repo "OmniGen2/OmniGen2" "$OMNIGEN_DIR"
fi

# -------------------- 2. FLUX.1-dev VAE -------------------------------------
FLUX_DIR="$TARGET_DIR/FLUX.1-dev"
if [ "$SKIP_FLUX" = "true" ]; then
    echo "[skip] FLUX.1-dev download skipped by flag."
else
    # Only grab the VAE sub-folder -- we do not need the text encoders or
    # transformer from FLUX. Saves ~25GB.
    if have_huggingface_cli; then
        mkdir -p "$FLUX_DIR"
        huggingface-cli download "black-forest-labs/FLUX.1-dev" \
            --local-dir "$FLUX_DIR" \
            --local-dir-use-symlinks False \
            --resume-download \
            --include "vae/*" "model_index.json"
    else
        python - <<PY
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="black-forest-labs/FLUX.1-dev",
    local_dir="$FLUX_DIR",
    local_dir_use_symlinks=False,
    resume_download=True,
    allow_patterns=["vae/*", "model_index.json"],
)
PY
    fi
fi

# -------------------- 3. Qwen2.5-VL-3B-Instruct -----------------------------
QWEN_DIR="$TARGET_DIR/Qwen2.5-VL-3B-Instruct"
if [ "$SKIP_QWEN" = "true" ]; then
    echo "[skip] Qwen2.5-VL-3B-Instruct download skipped by flag."
else
    download_repo "Qwen/Qwen2.5-VL-3B-Instruct" "$QWEN_DIR"
fi

echo
echo "=========================================================="
echo "All requested weights are now under: $TARGET_DIR"
echo
echo "Next step -- point your YAML at the local paths, e.g."
echo "  pretrained_vae_model_name_or_path:          $FLUX_DIR"
echo "  pretrained_text_encoder_model_name_or_path: $QWEN_DIR"
echo "  pretrained_model_path:                      $OMNIGEN_DIR"
echo "=========================================================="
