# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

export PYTHONPATH=$(pwd):$PYTHONPATH
# If you use viescore, make sure its directory is on PYTHONPATH, e.g.:
# export PYTHONPATH="./eval/gen/gedit/viescore:$PYTHONPATH"

# run this script at the root of the project folder
# pip install httpx==0.23.0
# pip install openai==1.87.0
# pip install datasets
# pip install megfile
# pip install utils
# pip install python-magic
# pip install autoawq autoawq-kernels --upgrade


export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

N_GPU=8
# Replace with your own paths.
OUTPUT_DIR="${OUTPUT_DIR:-outputs/gedit/bagel_sgt}"
MODEL_PATH="${MODEL_PATH:-ckpt/BAGEL-SGT}"

GEN_DIR="$OUTPUT_DIR/gen_image"
LOG_DIR="$OUTPUT_DIR/logs"

AZURE_ENDPOINT="https://azure_endpoint_url_you_use"  # set up the azure openai endpoint url
AZURE_OPENAI_KEY=""  # set up the azure openai key
N_GPT_PARALLEL=10


mkdir -p "$OUTPUT_DIR"
mkdir -p "$GEN_DIR"
mkdir -p "$LOG_DIR"


# # ----------------------------
# #    Download GEdit Dataset
# # ----------------------------
python -c "from datasets import load_dataset; dataset = load_dataset('stepfun-ai/GEdit-Bench')"
echo "Dataset Downloaded"


# # ---------------------
# #    Generate Images
# # ---------------------
# for ((i=0; i<$N_GPU; i++)); do
#     nohup python3 eval/gen/gedit/gen_images_gedit.py --model_path "$MODEL_PATH"  --output_dir "$GEN_DIR" --use-ema --shard_id $i --seed 5 --total_shards "$N_GPU" --device $i  2>&1 | tee "$LOG_DIR"/request_$(($N_GPU + i)).log &
# done

# wait
# echo "Image Generation Done"

python eval/gen/gedit/test_gedit_score.py \
    --save_path "$OUTPUT_DIR" \
    --backbone gpt4o \
    --gpt_keys "your_openai_api_key_here" \
    --max_workers 5

# # --------------------
# #    Print Results
# # --------------------
python eval/gen/gedit/calculate_statistics.py --save_path "$OUTPUT_DIR"  --language en
echo $OUTPUT_DIR
