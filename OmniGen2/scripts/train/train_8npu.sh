#!/bin/bash
# A script for single-node, 8-NPU distributed training

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

# 1. Navigate to the project root directory
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $(dirname $SHELL_FOLDER)
cd ../

# 2. Activate your NPU-enabled conda environment
# IMPORTANT: Make sure this is the environment with torch-npu installed, not a CUDA one.
source "$(dirname $(which conda))/../etc/profile.d/conda.sh"
conda activate omnigen2 # Placeholder name, please use your actual NPU environment name

# 3. Set parameters for the training script
experiment_name=ft

# 4. Use accelerate launch for 8 NPUs on the current machine
#    - All FSDP flags are removed.
#    - Multi-node flags are removed for simplicity.
#    - --num_processes is set to 8.
accelerate launch \
--num_processes=16 \
train_npu.py --config options/${experiment_name}.yml