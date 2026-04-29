#!/bin/bash
# Convert a training checkpoint directory to a HuggingFace-compatible layout.
#
# Usage:
#   bash scripts/trans2hf.sh
#
# Required environment / variables:
#   INPUT_CHECKPOINT_PATH : path to the training checkpoint directory (contains ema.safetensors etc.)
#   OUTPUT_HF_PATH        : where to write the converted HF weights
#   TEMPLATE_MODEL        : path to a reference HF model directory (e.g. BAGEL-7B-MoT)
#
# The paths below are placeholders. Replace them with your own absolute or
# relative paths before running.

INPUT_CHECKPOINT_PATH="${INPUT_CHECKPOINT_PATH:-checkpoints/experiment/step_0008000}"
OUTPUT_HF_PATH="${OUTPUT_HF_PATH:-outputs/hf_weights/experiment}"
TEMPLATE_MODEL="${TEMPLATE_MODEL:-ckpt/BAGEL-7B-MoT}"

# Print the command that will be executed, for easy debugging
echo "############################################################"
echo "### Processing: ${INPUT_CHECKPOINT_PATH}"
echo "### Output to:  ${OUTPUT_HF_PATH}"
echo "############################################################"

# Execute the Python conversion script
python scripts/trans2hf.py \
  --training_checkpoint_path "${INPUT_CHECKPOINT_PATH}" \
  --template_model_path "${TEMPLATE_MODEL}" \
  --output_path "${OUTPUT_HF_PATH}"
