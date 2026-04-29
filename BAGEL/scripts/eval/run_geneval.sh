# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

set -x

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
GPUS=8

# Replace these paths with your own (image output dir and HF model dir).
export output_path="${output_path:-outputs/geneval/bagel_sgt}"
export model_path="${model_path:-ckpt/BAGEL-SGT}"

# torchrun \
#     --nnodes=1 \
#     --node_rank=0 \
#     --nproc_per_node=$GPUS \
#     --master_addr=127.0.0.1 \
#     --master_port=12349 \
#     ./eval/gen/gen_images_mp.py \
#     --output_dir $output_path \
#     --metadata_file ./eval/gen/geneval/prompts/evaluation_metadata_long.jsonl \
#     --batch_size 1  \
#     --num_images 1 \
#     --resolution 1024 \
#     --max_latent_size 64 \
#     --model-path $model_path \
#     --seed 42\
#     --use_ema

## Calculate score
# torchrun \
#     --nnodes=1 \
#     --node_rank=0 \
#     --nproc_per_node=$GPUS \
#     --master_addr=127.0.0.1 \
#     --master_port=12343 \
#     ./eval/gen/geneval/evaluation/evaluate_images_mp.py \
#     $output_path \
#     --outfile $output_path/results.jsonl \
#     --model-path ./eval/gen/geneval/model

# Summarize score
python ./eval/gen/geneval/evaluation/summary_scores.py $output_path/results.jsonl
echo $output_path
