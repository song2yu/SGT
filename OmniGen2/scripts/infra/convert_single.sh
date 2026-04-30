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

# experiments_acc32_ug1_3 2000
# experiments_acc32_ug1_5 1500 2000 2500
# experiments_acc32_ug1_3_lr2e6 1400 1500 1600 1700 1800 1900 2000 2200 2500 2700 3000 3200 3300 3500
# UG12_lr2e6_bs32 1000 1500 2000 2500 3000 3500 4000
# UG12_lr4e6_bs48 1000 1500 2000 2500 3000 3500
# UG12_lr5e6_bs60  500 1000 1500 2000 2500 2750 3000 3250
# UG12_lr5e6_bs120 500 1000 1250 1500 1750 2000
EXPERIMENT_NAME='UG12_lr5e6_bs120'
CKPT_STEP='2000'
python scripts/infra/convert_single.py \
  --config_path ${PROJECT_ROOT}/experiments/${EXPERIMENT_NAME}/ft_panoptic/sft_panoptic.yml \
  --model_path ${PROJECT_ROOT}/experiments/${EXPERIMENT_NAME}/ft_panoptic/checkpoint-${CKPT_STEP}/model.safetensors \
  --save_path ${PROJECT_ROOT}/experiments/${EXPERIMENT_NAME}/ft_panoptic/checkpoint-${CKPT_STEP}/transformer
