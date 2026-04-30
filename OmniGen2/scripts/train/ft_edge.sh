#!/bin/bash
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $(dirname $SHELL_FOLDER)
cd ../

# Activate conda environment
source "$(dirname $(which conda))/../etc/profile.d/conda.sh"
conda activate omnigen2

# ================= Config =================
# true:  single-GPU debug mode (no FSDP, fast startup)
# false: multi-GPU training mode (8 GPUs, FSDP Hybrid Shard)
debug=false # false true

experiment_name=ft_edge
# ==========================================

RANK=0
MASTER_ADDR=1
MASTER_PORT=29500
WORLD_SIZE=1

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --rank=*)
            RANK="${1#*=}"
            shift
            ;;
        --master_addr=*)
            MASTER_ADDR="${1#*=}"
            shift
            ;;
        --master_port=*)
            MASTER_PORT="${1#*=}"
            shift
            ;;
        --world_size=*)
            WORLD_SIZE="${1#*=}"
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            shift
            ;;
    esac
done

echo "========================================"
echo "Experiment: $experiment_name"
echo "Debug Mode: $debug"
echo "RANK: $RANK, WORLD_SIZE: $WORLD_SIZE"
echo "========================================"

if [ "$debug" = "true" ]; then
    # ----------------------------------------
    # Scenario 1: single-GPU debug mode
    # ----------------------------------------
    echo "Entering [single-GPU debug mode]..."
    echo "1. Force only GPU 0 to be visible (CUDA_VISIBLE_DEVICES=0)"
    echo "2. Disable FSDP to avoid group_size errors"
    
    export CUDA_VISIBLE_DEVICES=0
    
    python -m accelerate.commands.launch \
    --num_processes=1 \
    --num_machines=1 \
    --main_process_port=$MASTER_PORT \
    train.py --config options/${experiment_name}.yml

else
    # ----------------------------------------
    # Scenario 2: full training mode (multi-GPU FSDP)
    # ----------------------------------------
    num_processes=$(($WORLD_SIZE * 8))
    echo "Entering [full training mode]..."
    echo "Number of processes (GPUs): $num_processes"
    echo "FSDP strategy enabled: HYBRID_SHARD_ZERO2"

    python -m accelerate.commands.launch \
    --machine_rank=$RANK \
    --main_process_ip=$MASTER_ADDR \
    --main_process_port=$MASTER_PORT \
    --num_machines=$WORLD_SIZE \
    --num_processes=$num_processes \
    --use_fsdp \
    --fsdp_offload_params false \
    --fsdp_sharding_strategy HYBRID_SHARD_ZERO2 \
    --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
    --fsdp_transformer_layer_cls_to_wrap OmniGen2TransformerBlock \
    --fsdp_state_dict_type FULL_STATE_DICT \
    --fsdp_forward_prefetch false \
    --fsdp_use_orig_params True \
    --fsdp_cpu_ram_efficient_loading false \
    --fsdp_sync_module_states True \
    train.py --config options/${experiment_name}.yml
fi