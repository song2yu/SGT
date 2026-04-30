#!/bin/bash
# run_gedit_eval.sh - GEdit Benchmark Evaluation for OmniGen2

SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $SHELL_FOLDER

# Resolve project root (scripts/infra/ -> project root)
PROJECT_ROOT=$(cd "${SHELL_FOLDER}/../.."; pwd)
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
MODEL_PATH="OmniGen2/OmniGen2"
# Model configuration
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${PROJECT_ROOT}/experiments/UG12_lr5e6_bs60/ft_panoptic/checkpoint-2500/}"


# Output directory
OUTPUT_DIR="./outputs_gedit/ft_panoptic_5k_seed3"

# Generation parameters
CFG_TEXT_SCALE=5.0
CFG_IMG_SCALE=2.0
NUM_STEPS=50
SEED=3

# Image size parameters
MAX_IMAGE_SIZE=1024
MIN_IMAGE_SIZE=512

# Distributed parameters
NUM_GPUS=8

# Create output directory
mkdir -p $OUTPUT_DIR

# Get random master port
MASTER_PORT=$(shuf -i 29500-29999 -n 1)
echo "Using master port: $MASTER_PORT"
echo "Output directory: $OUTPUT_DIR"
echo "Model path: $MODEL_PATH"

# Run evaluation
torchrun --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    ${PROJECT_ROOT}/eval/gen/gen_gedit.py \
    --transformer_path "${CHECKPOINT_DIR}/transformer/transformer" \
    --text_encoder_path "${CHECKPOINT_DIR}/transformer/text_encoder" \
    --model_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --cfg_text_scale $CFG_TEXT_SCALE \
    --cfg_img_scale $CFG_IMG_SCALE \
    --num_inference_steps $NUM_STEPS \
    --seed $SEED \
    --max_image_size $MAX_IMAGE_SIZE \
    --min_image_size $MIN_IMAGE_SIZE \
    --dtype bf16

echo "GEdit evaluation completed!"
echo "Results saved to: $OUTPUT_DIR"
