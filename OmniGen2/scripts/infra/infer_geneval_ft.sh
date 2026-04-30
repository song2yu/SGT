#!/bin/bash
# GenEval inference with a finetuned OmniGen2 checkpoint.
# pip install huggingface_hub
# pip install transformers -U

# Resolve project root from this script's location (scripts/infra/ -> project root)
SHELL_FOLDER=$(cd "$(dirname "$0")"; pwd)
PROJECT_ROOT=$(cd "${SHELL_FOLDER}/../.."; pwd)

export CUDA_VISIBLE_DEVICES=4,5,6,7,0,1,2,3
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Base model path
model_path="OmniGen2/OmniGen2"

# Finetuned checkpoint path; override via env var CHECKPOINT_DIR.
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${PROJECT_ROOT}/experiments/UG12_lr5e6_bs60/ft_panoptic/checkpoint-2500/}"
# Output directory (use distinct names per experiment)
# output_dir="${PROJECT_ROOT}/outputs_geneval/UG12_lr5e6_bs60_2500_seed0"
output_dir="${PROJECT_ROOT}/outputs_geneval/ft_hiomnigen2_2.5k_seed23"
MASTER_PORT=$(shuf -i 29500-29999 -n 1)
echo "Using master port: $MASTER_PORT"
echo "Output directory: $OUTPUT_DIR"
echo "Model path: $MODEL_PATH"

metadata_file="${GENEVAL_METADATA_FILE:-./eval/gen/geneval/prompts/evaluation_metadata.jsonl}"

torchrun --nproc_per_node=8 \
    --master_port=$MASTER_PORT \
    ${PROJECT_ROOT}/eval/gen/gen_images_mp_ft.py \
    --model_path $model_path \
    --transformer_path "${CHECKPOINT_DIR}/transformer/transformer" \
    --text_encoder_path "${CHECKPOINT_DIR}/transformer/text_encoder" \
    --metadata_file $metadata_file \
    --output_dir $output_dir \
    --num_images 1 \
    --num_inference_step 50 \
    --height 1024 \
    --width 1024 \
    --text_guidance_scale 4.0 \
    --seed 23 \
    --use_distributed
