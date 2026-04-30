#!/bin/bash
# Convert a trained checkpoint to HuggingFace format.

# Stop immediately on any error
set -e

# Print commands as they are executed (optional; useful for debugging)
set -x

export PYTHONPATH=$PYTHONPATH:.
# full: ft_semantic  ft_panoptic  ft_edge   ft_reca
# ft_panoptic
# python scripts/infra/convert_joint_ckpt_to_hf_format.py \
#   --config_path experiments/ft_reca_bs64/ft_reca.yml \
#   --model_path experiments/ft_reca_bs64/checkpoint-2000/ \
#   --save_path experiments/ft_reca_bs64/checkpoint-2000/transformer
# python scripts/infra/convert_joint_ckpt_to_hf_format.py \
#   --config_path experiments/ft_panoptic/ft_panoptic.yml \
#   --model_path experiments/ft_panoptic/checkpoint-2000/pytorch_model_fsdp.bin \
#   --save_path experiments/ft_panoptic/checkpoint-2000/transformer
python scripts/infra/convert_joint_ckpt_to_hf_format.py \
  --config_path experiments_full_gen/ft_panoptic/sft_panoptic.yml \
  --model_path experiments_full_gen/ft_panoptic/checkpoint-3000/pytorch_model_fsdp.bin \
  --save_path experiments_full_gen/ft_panoptic/checkpoint-3000/transformer


# experiments_full_gen/ft_panoptic/checkpoint-5000
# experiments/ft_reca_bs64/checkpoint-2000
