# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import io
import json
import os
from glob import glob
import tarfile
import tempfile
import pdb
import torch.utils.data
import pyarrow.parquet as pq

import random
from PIL import Image

import hashlib
from pathlib import Path
import traceback
import numpy as np  # Add import for numpy

from .data_utils import pil_img2rgb
from .distributed_iterable_dataset import DistributedIterableDataset
from .parquet_utils import get_parquet_data_paths, init_arrow_pf_fs

Image.MAX_IMAGE_PIXELS = 2_000_000_000

class Panoptic100KDataset(DistributedIterableDataset):
    def __init__(
        self, dataset_name, transform, vit_transform, tokenizer, data_dir_list, num_used_data, jsonl_path_list,
        local_rank=0, world_size=1, num_workers=8, data_status=None,
        cache_dir=None, from_huggingface=False, path_huggingface=None
    ):
        """
        Reconstruction Dataset processor for loading webdataset format data from tar packages.
        Supports caching mechanism to improve data loading efficiency.
        Also supports loading data directly from Hugging Face datasets.
        
        Args:
            from_huggingface: Whether to load data from Hugging Face instead of tar files
            path_huggingface: Path to the Hugging Face dataset, e.g., "brivangl/midjourney-v6-llava"
        """
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.transform = transform
        self.vit_transform = vit_transform
        self.tokenizer = tokenizer
        self.data_status = data_status
        self.from_huggingface = from_huggingface
        self.path_huggingface = path_huggingface
        self.jsonl_path = jsonl_path_list   
        self.data_dir = data_dir_list[0]


        from data.consts import get_segment_prompt_list
        import random
        self.prompt_templates = get_segment_prompt_list()
        # Initialize cache directory
        if cache_dir is None:
            self.cache_dir = os.path.join(tempfile.gettempdir(), "bagel_reconstruction_cache")
        else:
            self.cache_dir = cache_dir
            
        os.makedirs(self.cache_dir, exist_ok=True)
        print(f"Using cache directory: {self.cache_dir}")
        
        # Store cache path mappings for extracted tar files
        self.tar_cache_dirs = {}
        # Store filename to cache path mapping
        self.file_cache_paths = {}
        
        # Get data paths
        self.data_paths = self.get_data_paths(num_used_data)
        if self.data_paths:  # Only set epoch when data paths are not empty
            self.set_epoch()
        else:
            print(f"Warning: No data files found for {dataset_name}. Check your data_dir_list: {data_dir_list}")
        
    def read_jsonl_file(self, file_path):
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

    def read_txt_to_list(self, file_path):
        """
        Read a TXT file and return a list where each element is one line.

        Leading/trailing whitespace (including the trailing newline) is
        stripped from every line automatically.

        :param file_path: path to the TXT file (str)
        :return: list of lines (list of str)
        """

        # Check whether the file exists.
        if not os.path.exists(file_path):
            print(f"Error: file not found -> {file_path}")
            return None

        element_list = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # Use a list comprehension for efficiency.
                # line.strip() removes leading/trailing whitespace
                # (including \n, \r, spaces, ...).
                element_list = [line.strip() for line in f]

                # Optionally drop empty lines:
                # element_list = [line.strip() for line in f if line.strip()]

            return element_list

        except FileNotFoundError:
            # os.path.exists covers this in theory, but it's a good safety net.
            print(f"Error: file not found -> {file_path}")
            return None
        except Exception as e:
            print(f"Error while reading file: {e}")
            return None


    def get_data_paths(self, num_used_data):
        total_data = self.read_txt_to_list(self.jsonl_path[0])
        total_num = len(total_data)
        if total_num > num_used_data[0]:
            used_data = total_data[:num_used_data[0]]
        else:
            used_data = total_data
        self.all_items = used_data
        
        return self.all_items
        
    def _load_huggingface_data(self, num_used_data):
        """Load data from Hugging Face dataset"""
        all_items = []
        try:
            from datasets import load_dataset
            print(f"Loading dataset from {self.path_huggingface} with cache_dir {self.cache_dir}")
            dataset = load_dataset(self.path_huggingface, cache_dir=self.cache_dir)['train']
            print(f"Loaded {len(dataset)} samples from {self.path_huggingface}")
            
            # Create virtual items
            for idx in range(len(dataset)):
                all_items.append(('huggingface', f'sample_{idx}', [f'sample_{idx}']))
                
            print(f"Added {len(all_items)} samples from Hugging Face dataset {self.path_huggingface}")
            
            # Save the dataset for later use in __iter__
            self.hf_dataset = dataset
            
        except Exception as e:
            print(f"Error loading Hugging Face dataset: {e}")
            traceback.print_exc()
        
        # Limit data amount if needed
        if num_used_data and num_used_data[0] > 0 and num_used_data[0] < len(all_items):
            all_items = all_items[:num_used_data[0]]
            print(f"Limited to {len(all_items)} items")
          
        return all_items
                
    def _get_cached_image_path(self, tar_path, file_name):
        """Generate cached image path"""
        # Create a unique cache filename using tar path and internal filename
        tar_hash = hashlib.md5(tar_path.encode()).hexdigest()[:8]
        file_hash = hashlib.md5(file_name.encode()).hexdigest()[:8]
        cache_name = f"{tar_hash}_{file_hash}_{os.path.basename(file_name)}"
        return os.path.join(self.cache_dir, cache_name)

    def __iter__(self):
        """
        Iterator method for loading data from tar files or Hugging Face datasets
        Returns: A training sample for each iteration
        """
        # Get data paths and ID for current worker
        data_items_per_worker, worker_id = self.get_data_paths_per_worker()
        
        # Resume from checkpoint if data status is available
        if self.data_status is not None:
            item_start_id = self.data_status[worker_id] + 1
        else:
            item_start_id = 0
            
        transform_stride = self.transform.stride  # Image transform stride
        
        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming data at item#{item_start_id}"
        )
        
        # Infinite loop to enable dataset reuse
        
        while True:
            # Get data items starting from current index
            data_items_ = data_items_per_worker[item_start_id:]
            # for item_idx, (tar_path, base_name, files) in enumerate(data_items_, start=item_start_id):
            for item_idx, data_item in enumerate(data_items_, start=item_start_id):
                try:
                    image_file = self.data_dir + data_item
                    edited_file = image_file.replace('train2017', 'annotations/panoptic_train2017').replace('jpg', 'png') 
                    try:
                        image = pil_img2rgb(Image.open(image_file))
                        edited_image = pil_img2rgb(Image.open(edited_file))
                        edited_image = edited_image.resize(image.size)
                    except Exception as e:
                        print(f"Error loading image {image_file}: {e}")
                        continue
                    
                    # Directly randomly select a prompt from the template, not relying on JSON metadata
                    prompt = random.choice(self.prompt_templates)
                    # prompt = data_item['generation_prompt']
                    # print(prompt)
                    # Apply image transformation
                    image_tensor = self.transform(image) # 4096
                    eidted_tensor = self.transform(edited_image)
                    vit_image_tensor = self.vit_transform(image) # 256
                    
                    num_tokens = 0
                    height, width = image_tensor.shape[1:]
                    num_tokens += width * height // transform_stride ** 2
                    height_vit, width_vit = vit_image_tensor.shape[1:]
                    num_tokens += width_vit * height_vit // self.vit_transform.stride ** 2
                    # Encode prompt text as token sequence
                    
                    caption_token = self.tokenizer.encode(prompt)
                    
                    # Initialize sequence plan and text ID list
                    sequence_plan, text_ids_list = [], []
                    text_ids = caption_token
                    
                    # Update token count by adding text token count
                    num_tokens += len(caption_token)
                    # print(f'after caption_token: {num_tokens}, caption_token:{len(caption_token)}')
                    text_ids_list.append(text_ids)

                    sequence_plan.append({
                        'type': 'vit_image',
                        'enable_cfg': 0,
                        'loss': 0,
                        'special_token_loss': 0,
                        'special_token_label': None,
                    })
                    
                    # Add text processing plan (prompt part)
                    sequence_plan.append({
                        'type': 'text',
                        'enable_cfg': 1,
                        'loss': 0,
                        'special_token_loss': 0,
                        'special_token_label': None,
                    })
                    
                    # Add image processing plan (reconstruction target)
                    sequence_plan.append({
                        'type': 'vae_image',
                        'enable_cfg': 0,
                        'loss': 1,
                        'special_token_loss': 0,
                        'special_token_label': None,
                    })
                    
                    # Build sample dictionary
                    sample = dict(
                        image_tensor_list=[vit_image_tensor, eidted_tensor, image_tensor], # 
                        text_ids_list=text_ids_list,
                        num_tokens=num_tokens,
                        sequence_plan=sequence_plan,
                        data_indexes={
                            "data_indexes": item_idx,
                            "worker_id": worker_id,
                            "dataset_name": self.dataset_name,
                        }
                    )
                    
                    yield sample  # Return a sample
                
                except Exception as e:
                    print(f"Error processing item {image_file}: {e}")
                    traceback.print_exc()
                    continue
            
            # Reset start ID after processing all data (restart)
            item_start_id = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")
            
    def cleanup_cache(self):
        """Clean up temporary files in cache directory"""
        if os.path.exists(self.cache_dir):
            print(f"Cleaning up cache directory: {self.cache_dir}")
            for cached_file in os.listdir(self.cache_dir):
                try:
                    os.remove(os.path.join(self.cache_dir, cached_file))
                except Exception as e:
                    print(f"Error removing cached file {cached_file}: {e}")