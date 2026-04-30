#!/bin/bash
# GenEval inference with the base OmniGen2 model.
# pip install huggingface_hub
# pip install transformers -U

# Resolve project root from this script's location (scripts/infra/ -> project root)
SHELL_FOLDER=$(cd "$(dirname "$0")"; pwd)
PROJECT_ROOT=$(cd "${SHELL_FOLDER}/../.."; pwd)

# Set PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

model_path="OmniGen2/OmniGen2"
output_dir="${PROJECT_ROOT}/outputs_geneval/OmniGen2_12"
# Path to the GenEval prompt metadata file; override via env var or CLI as needed
metadata_file="${GENEVAL_METADATA_FILE:-./eval/gen/geneval/prompts/evaluation_metadata.jsonl}"

torchrun --nproc_per_node=8 ${PROJECT_ROOT}/eval/gen/gen_images_mp.py \
    --model_path $model_path \
    --metadata_file $metadata_file \
    --output_dir $output_dir \
    --num_images 1 \
    --num_inference_step 50 \
    --height 1024 \
    --width 1024 \
    --text_guidance_scale 4.0 \
    --seed 5 \
    --use_distributed
