#!/bin/bash
# Convert a single-file checkpoint (model.safetensors) to HuggingFace format.

# Stop immediately on any error
set -e

# Print commands as they are executed (optional; useful for debugging)
set -x

# Resolve project root from this script's location (scripts/infra/ -> project root)
SHELL_FOLDER=$(cd "$(dirname "$0")"; pwd)
PROJECT_ROOT=$(cd "${SHELL_FOLDER}/../.."; pwd)

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
# full: ft_semantic  ft_panoptic  ft_edge   ft_reca


# UG12_lr5e6_bs60  2000 2500

EXPERIMENT_NAME='UG12_lr5e6_bs60'
CKPT_STEP='2000'
python scripts/infra/convert_single.py \
  --config_path ${PROJECT_ROOT}/experiments/${EXPERIMENT_NAME}/ft_panoptic/sft_panoptic.yml \
  --model_path ${PROJECT_ROOT}/experiments/${EXPERIMENT_NAME}/ft_panoptic/checkpoint-${CKPT_STEP}/model.safetensors \
  --save_path ${PROJECT_ROOT}/experiments/${EXPERIMENT_NAME}/ft_panoptic/checkpoint-${CKPT_STEP}/transformer
