IMAGE_ROOT_PATH=${1:-"outputs/dpg/bagel_sgt"}
RESOLUTION=${2:-1024}
# 1024 82.8609
# 512 82.8586
# official 83.9714
PIC_NUM=${PIC_NUM:-4}
GPU_IDS=${GPU_IDS:-"0,1,2,3,4,5,6,7"}  
# GPU_IDS=${GPU_IDS:-"0"}  # debug: download model

export CUDA_VISIBLE_DEVICES=$GPU_IDS
NUM_GPUS=$(echo $GPU_IDS | tr ',' '\n' | wc -l)
PROCESSES=$NUM_GPUS
PORT=${PORT:-29505}

echo "Use GPU: $GPU_IDS ( $NUM_GPUS GPUs )"
echo "Start $PROCESSES processes"

accelerate launch --num_machines 1 --num_processes $PROCESSES --mixed_precision "fp16" --main_process_port $PORT \
  compute_dpg_bench.py \
  --image-root-path $IMAGE_ROOT_PATH \
  --resolution $RESOLUTION \
  --pic-num $PIC_NUM \
  --vqa-model mplug
  #  --multi_gpu

# bash dpg_bench/dist_eval.sh $YOUR_IMAGE_PATH $RESOLUTION
