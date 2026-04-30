#!/bin/bash
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $(dirname $SHELL_FOLDER)
cd ../

debug=false # false
experiment_name=sft_panoptic

RANK=0
MASTER_ADDR=127.0.0.1
MASTER_PORT=29500
WORLD_SIZE=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --rank=*) RANK="${1#*=}"; shift ;;
        --master_addr=*) MASTER_ADDR="${1#*=}"; shift ;;
        --master_port=*) MASTER_PORT="${1#*=}"; shift ;;
        --world_size=*) WORLD_SIZE="${1#*=}"; shift ;;
        *) echo "Unknown argument: $1"; shift ;;
    esac
done

num_processes=$(($WORLD_SIZE * 8))

echo "========================================"
echo "Data parallel mode: ${num_processes} GPUs"
echo "Full model per GPU, data is auto-sharded"
echo "========================================"

if [ "$debug" = "true" ]; then
    export CUDA_VISIBLE_DEVICES=0
    python -m accelerate.commands.launch \
        --num_processes=1 \
        train.py --config options/${experiment_name}.yml
else
    python -m accelerate.commands.launch \
        --machine_rank=$RANK \
        --main_process_ip=$MASTER_ADDR \
        --main_process_port=$MASTER_PORT \
        --num_machines=$WORLD_SIZE \
        --num_processes=$num_processes \
        --multi_gpu \
        train.py --config options/${experiment_name}.yml
fi