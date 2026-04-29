# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import os
from copy import deepcopy
import pdb
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
from pathlib import Path
import warnings
import json

def read_jsonl_file(file_path):
    """
    read JSON Lines
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                entry = json.loads(line)
                data.append(entry)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
    return data


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

model_path = "ckpt/BAGEL-7B-MoT"  # Download from https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT

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



max_mem_per_gpu = "80GiB"  # Modify it according to your GPU setting. On an A100, 80 GiB is sufficient to load on a single GPU.

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

# Thanks @onion-liu: https://github.com/ByteDance-Seed/Bagel/pull/8
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

import random
import numpy as np

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



inference_hyper=dict(
    cfg_text_scale=4.0,
    cfg_img_scale=2.0,
    cfg_interval=[0.0, 1.0],
    timestep_shift=3.0,
    num_timesteps=50,
    cfg_renorm_min=0.0,
    cfg_renorm_type="text_channel",
)


test_file = 'data/omniedit-reca/multiple_repaint_info_pro.jsonl'
samples = read_jsonl_file(test_file)
img_dir = 'data/omniedit/images/'
output_dir = Path('outputs/bagel_referring-5k-repaint-wo-vae')

if output_dir.exists():
    if not output_dir.is_dir():
        warnings.warn(f"Warning: Path '{output_dir}' exists but is not a directory.")
    else:
        warnings.warn(f"Warning: Directory '{output_dir}' already exists.")
else:
    output_dir.mkdir(parents=True)
    print(f"Directory '{output_dir}' created successfully.")

ids = 0
for item in samples:
    prompt = item['spatial_instruction']
    img_name = item['omni_edit_id'] + '_src.png'
    img_source = img_dir + img_name
    image = Image.open(img_source)
    output_image_path = os.path.join(output_dir, img_name)

    output_dict = inferencer(image=image, text=prompt, image_understanding_to_image=True, image_shapes=(527,816), think=True, **inference_hyper)

    vis_images = [to_tensor(output_dict['image']) * 2 - 1]
    output_image = create_collage(vis_images)
    output_image.save(output_image_path)
    ids += 1
    print(f'current iter:{ids}, prompt:{prompt}')



