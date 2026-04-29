#!/bin/bash
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

set -x
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
GPUS=8

# Replace these with your own output/model paths.
export output_path="${output_path:-outputs/dpg/bagel_sgt}"
export model_path="${model_path:-ckpt/BAGEL-SGT}"
export PYTHONPATH=$(pwd):$PYTHONPATH

mkdir -p $output_path

torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node=$GPUS \
    --master_addr=127.0.0.1 \
    --master_port=12345 \
    ./eval/gen/gen_images_dpg.py \
    --outdir $output_path \
    --prompts_file ./eval/gen/dpgbench/prompts.json \
    --batch_size 1 \
    --num_images 1 \
    --cfg_scale 3.0 \
    --resolution 1024 \
    --num_timesteps 50 \
    --max_latent_size 64 \
    --model-path $model_path \
    --seed 15 \
    --use-ema
