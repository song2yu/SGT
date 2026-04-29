# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
from copy import deepcopy
import pdb
import json
from typing import (
    Any,
    AsyncIterable,
    Callable,
    Dict,
    Generator,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)

import requests
from io import BytesIO

from PIL import Image
import torch
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights

from data.transforms import ImageTransform
from data.data_utils import pil_img2rgb, add_special_tokens
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from torchvision.transforms.functional import to_pil_image, to_tensor
from modeling.qwen2 import Qwen2Tokenizer
from modeling.bagel.qwen2_navit import NaiveCache
from modeling.autoencoder import load_ae
from safetensors.torch import load_file
from tqdm import tqdm
import random
import numpy as np


def create_collage(images: List[torch.Tensor]) -> Image.Image:
    """Create a horizontal collage from a list of images."""
    max_height = max(img.shape[-2] for img in images)
    total_width = sum(img.shape[-1] for img in images)
    canvas = torch.zeros((3, max_height, total_width), device=images[0].device)
    
    current_x = 0
    for img in images:
        h, w = img.shape[-2:]
        canvas[:, :h, current_x:current_x+w] = img * 0.5 + 0.5
        current_x += w
    
    return to_pil_image(canvas)


def set_seed(seed):
    """Seed every relevant RNG."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_prompts(jsonl_path):
    """Read prompts from a JSON Lines file."""
    prompts = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                prompts.append(data)
    return prompts


# ============ Model loading. ============
model_path = "ckpt/BAGEL-7B-MoT"

# LLM config preparing
llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
llm_config.qk_norm = True
llm_config.tie_word_embeddings = False
llm_config.layer_module = "Qwen2MoTDecoderLayer"

# ViT config preparing
vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
vit_config.rope = False
vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

# VAE loading
vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

# Bagel config preparing
config = BagelConfig(
    visual_gen=True,
    visual_und=True,
    llm_config=llm_config, 
    vit_config=vit_config,
    vae_config=vae_config,
    vit_max_num_patch_per_side=70,
    connector_act='gelu_pytorch_tanh',
    latent_patch_size=2,
    max_latent_size=64,
)

with init_empty_weights():
    language_model = Qwen2ForCausalLM(llm_config)
    vit_model      = SiglipVisionModel(vit_config)
    model          = Bagel(language_model, vit_model, config)
    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

# Tokenizer Preparing
tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

# Image Transform Preparing
vae_transform = ImageTransform(1024, 512, 16)
vit_transform = ImageTransform(980, 224, 14)

max_mem_per_gpu = "80GiB"

device_map = infer_auto_device_map(
    model,
    max_memory={5: max_mem_per_gpu},
    no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
)
print(device_map)

same_device_modules = [
    'language_model.model.embed_tokens',
    'time_embedder',
    'latent_pos_embed',
    'vae2llm',
    'llm2vae',
    'connector',
    'vit_pos_embed'
]

if torch.cuda.device_count() == 1:
    first_device = device_map.get(same_device_modules[0], "cuda:0")
    for k in same_device_modules:
        if k in device_map:
            device_map[k] = first_device
        else:
            device_map[k] = "cuda:0"
else:
    first_device = device_map.get(same_device_modules[0])
    for k in same_device_modules:
        if k in device_map:
            device_map[k] = first_device

model = load_checkpoint_and_dispatch(
    model,
    checkpoint=os.path.join('ckpt/BAGEL-SGT/ema.safetensors'),
    device_map="auto",
    offload_buffers=False,
    dtype=torch.bfloat16,
    force_hooks=True,
)

model = model.eval()
print('Model loaded')

from inferencer import InterleaveInferencer

inferencer = InterleaveInferencer(
    model=model, 
    vae_model=vae_model, 
    tokenizer=tokenizer, 
    vae_transform=vae_transform, 
    vit_transform=vit_transform, 
    new_token_ids=new_token_ids
)

# ============ Batch generation configuration. ============
# I/O configuration.
PROMPTS_FILE = "scripts/prompt.jsonl"  # path to the prompts jsonl file
OUTPUT_DIR = "./generated_images"  # output directory
SEEDS = [42, 123, 456, 789]  # four different seeds
IMAGE_SIZE = (1024, 1024)  # output image size

# Inference hyper-parameters.
inference_hyper = dict(
    cfg_text_scale=4.0,
    cfg_img_scale=1.0,
    cfg_interval=[0.4, 1.0],
    timestep_shift=3.0,
    num_timesteps=50,
    cfg_renorm_min=1.0,
    cfg_renorm_type="global",
)

# ============ Begin batch generation. ============
# Create the output directory.
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load prompts.
prompts = load_prompts(PROMPTS_FILE)
print(f"Loaded {len(prompts)} prompts")
print(f"Will generate {len(prompts) * len(SEEDS)} images in total")

# Persist generation records.
generation_log = []

# Iterate over every prompt.
total_iterations = len(prompts) * len(SEEDS)
pbar = tqdm(total=total_iterations, desc="Generating images")

for prompt_data in prompts:
    prompt_id = prompt_data['id']
    prompt_text = prompt_data['prompt']
    
    for seed in SEEDS:
        # Seed RNGs.
        set_seed(seed)
        
        # Generate images.
        try:
            output_dict = inferencer(
                text=prompt_text, 
                **inference_hyper, 
                image_understanding_to_image=False, 
                image_shapes=IMAGE_SIZE
            )
            
            # Save the image.
            if output_dict.get('image'):
                # File name pattern: id{id}_seed{seed}.png
                filename = f"id{prompt_id:03d}_seed{seed}.png"
                filepath = os.path.join(OUTPUT_DIR, filename)
                output_dict['image'].save(filepath)
                
                # Record the log entry.
                generation_log.append({
                    "id": prompt_id,
                    "seed": seed,
                    "prompt": prompt_text,
                    "filepath": filepath,
                    "status": "success"
                })
            else:
                generation_log.append({
                    "id": prompt_id,
                    "seed": seed,
                    "prompt": prompt_text,
                    "filepath": None,
                    "status": "failed - no image output"
                })
                
        except Exception as e:
            print(f"\nError generating id={prompt_id}, seed={seed}: {e}")
            generation_log.append({
                "id": prompt_id,
                "seed": seed,
                "prompt": prompt_text,
                "filepath": None,
                "status": f"failed - {str(e)}"
            })
        
        pbar.update(1)
        pbar.set_postfix({"id": prompt_id, "seed": seed})

pbar.close()

# ============ Save the generation log. ============
log_path = os.path.join(OUTPUT_DIR, "generation_log.jsonl")
with open(log_path, 'w', encoding='utf-8') as f:
    for record in generation_log:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')

# ============ Log summary statistics. ============
success_count = sum(1 for r in generation_log if r['status'] == 'success')
fail_count = len(generation_log) - success_count

print("\n" + "=" * 80)
print("Generation Complete!")
print("=" * 80)
print(f"Total prompts: {len(prompts)}")
print(f"Seeds per prompt: {len(SEEDS)}")
print(f"Total images attempted: {total_iterations}")
print(f"Successful: {success_count}")
print(f"Failed: {fail_count}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Generation log: {log_path}")
print("=" * 80)