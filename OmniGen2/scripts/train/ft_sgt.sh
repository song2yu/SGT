#!/bin/bash
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $(dirname $SHELL_FOLDER)
cd ../

# Activate conda env if needed
# source "$(dirname $(which conda))/../etc/profile.d/conda.sh"
# conda activate omnigen2

# ================= Config =================
# true:  single-GPU debug mode (no FSDP, fast startup)
# false: multi-GPU training mode (8 GPUs, FSDP Hybrid Shard)
debug=true 

experiment_name=sft_panoptic
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

# Disable hf_xet backend to avoid permission errors in restricted environments.
export HF_HUB_DISABLE_XET=1

# Redirect HF cache to a writable local dir under the project.
# Run shells/download_pretrained.sh first to populate pretrained_models/.
OMNIGEN2_HF_CACHE="$(cd $(dirname $0)/../..; pwd)/pretrained_models/.hf_cache"
unset TRANSFORMERS_CACHE HF_DATASETS_CACHE HF_HUB_CACHE HUGGINGFACE_HUB_CACHE \
      HF_HOME XDG_CACHE_HOME 2>/dev/null || true
export HF_HOME="$OMNIGEN2_HF_CACHE"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
# Run fully offline: rely only on local weights/json files.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"
echo "[env] HF_HOME            = $HF_HOME"
echo "[env] HF_DATASETS_CACHE  = $HF_DATASETS_CACHE"

# Data roots consumed by OmniGen2TrainDataset.
# Override PUBLIC_DATA_ROOT (or the per-task vars below) to match your machine.
PUBLIC_DATA_ROOT=${PUBLIC_DATA_ROOT:-/path/to/your/datasets}
export OMNIGEN2_SFT_IMAGE_ROOT="${PUBLIC_DATA_ROOT}/LLaVA-OneVision-SGT/llava_onevision_balanced/images"
: "${OMNIGEN2_COCO_ROOT:=${PUBLIC_DATA_ROOT}/coco/train2017}"
: "${OMNIGEN2_OMNIEDIT_ROOT:=${PUBLIC_DATA_ROOT}/omniedit/images}"
export OMNIGEN2_COCO_ROOT OMNIGEN2_OMNIEDIT_ROOT
# <DATA_ROOT> placeholder in data_configs/*.yml is resolved via this var.
export OMNIGEN2_DATA_ROOT="${PUBLIC_DATA_ROOT}"

echo "[env] OMNIGEN2_SFT_IMAGE_ROOT = $OMNIGEN2_SFT_IMAGE_ROOT"
echo "[env] OMNIGEN2_COCO_ROOT      = $OMNIGEN2_COCO_ROOT"
echo "[env] OMNIGEN2_OMNIEDIT_ROOT  = $OMNIGEN2_OMNIEDIT_ROOT"
echo "[env] OMNIGEN2_DATA_ROOT      = $OMNIGEN2_DATA_ROOT"

# Local Qwen2.5-VL-3B processor path (built by shells/download_pretrained.sh).
export OMNIGEN2_QWEN_PROCESSOR_PATH="$(cd $(dirname $0)/../..; pwd)/pretrained_models/Qwen2.5-VL-3B-Instruct"
echo "[env] OMNIGEN2_QWEN_PROCESSOR_PATH = $OMNIGEN2_QWEN_PROCESSOR_PATH"

# Experiment trackers OFF by default. Set ENABLE_WANDB=1 to re-enable wandb.
if [ "${ENABLE_WANDB:-0}" = "1" ]; then
    echo "[logger] wandb ENABLED (ENABLE_WANDB=1)"
    unset WANDB_DISABLED WANDB_MODE
else
    echo "[logger] wandb DISABLED (set ENABLE_WANDB=1 to re-enable)"
    export WANDB_DISABLED=true
    export WANDB_MODE=disabled
    export WANDB_SILENT=true
fi

if [ "$debug" = "true" ]; then
    # Single-GPU debug mode
    echo "Entering [single-GPU debug mode]..."
    echo "1. Force only GPU 0 to be visible (CUDA_VISIBLE_DEVICES=0)"
    echo "2. Disable FSDP to avoid group_size errors"
    
    export CUDA_VISIBLE_DEVICES=0
    # Set DEBUG_CUDA=1 to make CUDA kernels synchronous (slower but easier to debug).
    if [ "${DEBUG_CUDA:-0}" = "1" ]; then
        echo "[debug] CUDA_LAUNCH_BLOCKING=1 (synchronous kernels)"
        export CUDA_LAUNCH_BLOCKING=1
        export TORCH_USE_CUDA_DSA=1
    fi
    
    python -m accelerate.commands.launch \
    --num_processes=1 \
    --num_machines=1 \
    --main_process_port=$MASTER_PORT \
    train.py --config options/${experiment_name}.yml

else
    # Full training mode (multi-GPU FSDP)
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
