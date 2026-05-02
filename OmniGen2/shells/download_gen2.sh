#!/bin/bash
set -e

# Disable hf_xet to avoid permission errors in restricted environments
export HF_HUB_DISABLE_XET=1

# Uncomment the following line to use HF-Mirror (for users in mainland China)
# export HF_ENDPOINT=https://hf-mirror.com

REPO_ID="OmniGen2/OmniGen2"
TARGET_DIR="./pretrained_models/OmniGen2"

mkdir -p "${TARGET_DIR}"

huggingface-cli download \
    "${REPO_ID}" \
    --local-dir "${TARGET_DIR}" \
    --local-dir-use-symlinks False \
    --resume-download

echo "✅ Done"