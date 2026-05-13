#!/bin/bash
# 8-GPU single-node training launcher: FSDP ZeRO-2 (HYBRID_SHARD_ZERO2).
# FSDP is used because the joint model runs different sub-networks per
# micro-batch (gen = DiT + Qwen, sft = Qwen only), which DDP cannot handle.
# ZeRO-2 shards gradients + optim states but keeps full parameter replicas.

# Resolve the actual script path whether this file is executed via ``bash
# script.sh`` or sourced via ``source script.sh``.
SCRIPT_PATH="${BASH_SOURCE[0]}"
SHELL_FOLDER=$(cd "$(dirname "$SCRIPT_PATH")"; pwd)
PROJECT_ROOT=$(cd "$SHELL_FOLDER/../.."; pwd)
cd "$PROJECT_ROOT"

# --- 1. Activate the OmniGen2 venv ------------------------------------------
if [ -f activate_env.sh ]; then
    # shellcheck disable=SC1091
    source activate_env.sh
else
    echo "WARN: activate_env.sh not found at $(pwd); continuing with the"
    echo "       current python ($(which python))" >&2
fi

# ================= Config =================
experiment_name=sft_panoptic

# Auto-detect number of visible GPUs; override via NUM_GPUS=N.
if [ -z "${NUM_GPUS}" ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        NUM_GPUS=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || echo 0)
    fi
    NUM_GPUS=${NUM_GPUS:-0}
    if [ "$NUM_GPUS" -le 0 ]; then
        echo "ERROR: could not detect any GPU via nvidia-smi." >&2
        echo "       Export NUM_GPUS=N explicitly if you know what you're doing." >&2
        exit 1
    fi
fi
# ==========================================

# Multi-node plumbing (single-node by default).
RANK=0
MASTER_ADDR=127.0.0.1
MASTER_PORT=29500
WORLD_SIZE=1

# Parse named arguments.
while [[ $# -gt 0 ]]; do
    case "$1" in
        --rank=*)         RANK="${1#*=}"; shift ;;
        --master_addr=*)  MASTER_ADDR="${1#*=}"; shift ;;
        --master_port=*)  MASTER_PORT="${1#*=}"; shift ;;
        --world_size=*)   WORLD_SIZE="${1#*=}"; shift ;;
        --num_gpus=*)     NUM_GPUS="${1#*=}"; shift ;;
        *)                echo "Unknown argument: $1"; shift ;;
    esac
done

num_processes=$((WORLD_SIZE * NUM_GPUS))

echo "========================================"
echo "Experiment        : $experiment_name"
echo "Launcher          : FSDP ZeRO-2 (HYBRID_SHARD_ZERO2)"
echo "GPUs per node     : $NUM_GPUS  (auto-detected from nvidia-smi -L)"
echo "WORLD_SIZE (nodes): $WORLD_SIZE"
echo "Total processes   : $num_processes"
echo "RANK (this node)  : $RANK"
echo "----------------------------------------"
echo "Detected GPUs:"
nvidia-smi -L 2>&1 | sed 's/^/  /'
echo "========================================"

# --- HF workaround knobs (identical to ft_panoptic.sh) ----------------------
export HF_HUB_DISABLE_XET=1

export HF_HOME="$PROJECT_ROOT/pretrained_models/.hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"

# --- Data roots consumed by OmniGen2TrainDataset -----------------------------
# Override PUBLIC_DATA_ROOT (or the per-task vars below) to match your machine.
PUBLIC_DATA_ROOT=${PUBLIC_DATA_ROOT:-/path/to/your/datasets}
export OMNIGEN2_SFT_IMAGE_ROOT="${PUBLIC_DATA_ROOT}/LLaVA-OneVision-SGT/llava_onevision_balanced/images"
: "${OMNIGEN2_COCO_ROOT:=${PUBLIC_DATA_ROOT}/coco/train2017}"
: "${OMNIGEN2_OMNIEDIT_ROOT:=${PUBLIC_DATA_ROOT}/omniedit/images}"
export OMNIGEN2_COCO_ROOT OMNIGEN2_OMNIEDIT_ROOT
export OMNIGEN2_DATA_ROOT="${PUBLIC_DATA_ROOT}"

echo "[env] OMNIGEN2_SFT_IMAGE_ROOT = $OMNIGEN2_SFT_IMAGE_ROOT"
echo "[env] OMNIGEN2_COCO_ROOT      = $OMNIGEN2_COCO_ROOT"
echo "[env] OMNIGEN2_OMNIEDIT_ROOT  = $OMNIGEN2_OMNIEDIT_ROOT"
echo "[env] OMNIGEN2_DATA_ROOT      = $OMNIGEN2_DATA_ROOT"

export OMNIGEN2_QWEN_PROCESSOR_PATH="$PROJECT_ROOT/pretrained_models/Qwen2.5-VL-3B-Instruct"
echo "[env] OMNIGEN2_QWEN_PROCESSOR_PATH = $OMNIGEN2_QWEN_PROCESSOR_PATH"

# --- Logger default OFF (set ENABLE_WANDB=1 to turn it on) ------------------
if [ "${ENABLE_WANDB:-0}" = "1" ]; then
    echo "[logger] wandb ENABLED (ENABLE_WANDB=1)"
    unset WANDB_DISABLED WANDB_MODE
else
    echo "[logger] wandb DISABLED (set ENABLE_WANDB=1 to re-enable)"
    export WANDB_DISABLED=true
    export WANDB_MODE=disabled
    export WANDB_SILENT=true
fi

# Optional CUDA-side debug.
if [ "${DEBUG_CUDA:-0}" = "1" ]; then
    echo "[debug] CUDA_LAUNCH_BLOCKING=1 (synchronous kernels)"
    export CUDA_LAUNCH_BLOCKING=1
    export TORCH_USE_CUDA_DSA=1
fi

# --- NCCL safe defaults ------------------------------------------------------
# Disable features that have been observed to hang/misbehave on some clusters.
# Override any of these via the usual env-var mechanism to re-enable.
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}
export TORCH_NCCL_AVOID_RECORD_STREAMS=${TORCH_NCCL_AVOID_RECORD_STREAMS:-1}
export TORCH_NCCL_BLOCKING_WAIT=${TORCH_NCCL_BLOCKING_WAIT:-1}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
export TORCH_NCCL_TIMEOUT_SECONDS=${TORCH_NCCL_TIMEOUT_SECONDS:-300}

echo "[nccl] NCCL_NVLS_ENABLE=$NCCL_NVLS_ENABLE"
echo "[nccl] TORCH_NCCL_BLOCKING_WAIT=$TORCH_NCCL_BLOCKING_WAIT"
echo "[nccl] TORCH_NCCL_TIMEOUT_SECONDS=$TORCH_NCCL_TIMEOUT_SECONDS"

# --- Override global_batch_size for multi-GPU -------------------------------
# YAML default is tuned for 1 GPU (global_batch_size=9). train.py asserts
#     batch_size * grad_accum * num_processes == global_batch_size
# With batch_size=3 and grad_accum=3 from the YAML, we need
#     global_batch_size = 3 * 3 * num_processes = 9 * num_processes
global_batch_size=$((9 * num_processes))
echo "[override] global_batch_size = $global_batch_size"
echo "           (batch_size=3 * grad_accum=3 * num_processes=$num_processes)"

# --- Launch: FSDP ZeRO-2 (HYBRID_SHARD_ZERO2) -------------------------------
# We wrap the DiT and the text encoder as two SEPARATE FSDP modules instead
# of a single composite model. This avoids hangs observed when accelerate
# auto-walks a 7.7B composite module on the first forward.
echo "Entering [multi-GPU FSDP ZeRO-2 mode, per-submodel wrap]..."
python -m accelerate.commands.launch \
    --machine_rank=$RANK \
    --main_process_ip=$MASTER_ADDR \
    --main_process_port=$MASTER_PORT \
    --num_machines=$WORLD_SIZE \
    --num_processes=$num_processes \
    --mixed_precision=bf16 \
    --dynamo_backend=no \
    --use_fsdp \
    --fsdp_offload_params false \
    --fsdp_sharding_strategy HYBRID_SHARD_ZERO2 \
    --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
    --fsdp_transformer_layer_cls_to_wrap "OmniGen2TransformerBlock,Qwen2_5_VLDecoderLayer,Qwen2_5_VLVisionBlock" \
    --fsdp_state_dict_type FULL_STATE_DICT \
    --fsdp_forward_prefetch false \
    --fsdp_use_orig_params False \
    --fsdp_cpu_ram_efficient_loading false \
    --fsdp_sync_module_states False \
    train.py \
        --config options/${experiment_name}.yml \
        --global_batch_size $global_batch_size
