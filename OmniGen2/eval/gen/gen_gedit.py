# Copyright 2025 OmniGen2 Team
# GEdit Benchmark Evaluation for OmniGen2

import dotenv
dotenv.load_dotenv(override=True)

import os
import sys
import glob
import argparse
import random
import datetime
import torch
import torch.distributed as dist
import numpy as np
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from safetensors.torch import load_file

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

if not torch.cuda.is_available():
    import torch_npu
    from torch_npu.contrib import transfer_to_npu

from diffusers.hooks import apply_group_offloading
from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel


def set_seeds(seed):
    """Set random seeds for reproducibility."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_distributed():
    """Initialize distributed training environment."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank
    else:
        return 0, 1, 0


def cleanup_distributed():
    """Clean up distributed environment."""
    if dist.is_initialized():
        try:
            dist.destroy_process_group()
        except:
            pass


def load_text_encoder_weights(pipeline, text_encoder_path: str, rank: int = 0):
    """Load finetuned text encoder (mllm) weights with key remapping."""
    if rank == 0:
        print(f"Loading finetuned text encoder weights from: {text_encoder_path}")
    
    state_dict = {}
    files = sorted(glob.glob(os.path.join(text_encoder_path, "*.safetensors")))
    if not files:
        files = sorted(glob.glob(os.path.join(text_encoder_path, "*.bin")))
    
    if not files:
        raise FileNotFoundError(f"No weights found in {text_encoder_path}")

    if rank == 0:
        print(f"Reading {len(files)} weight files...")
    
    for f in files:
        if f.endswith(".safetensors"):
            state_dict.update(load_file(f))
        else:
            state_dict.update(torch.load(f, map_location="cpu"))

    if rank == 0:
        print("Remapping keys (Adding 'model.' prefix)...")
    
    new_state_dict = {}
    for k, v in state_dict.items():
        if not k.startswith("model."):
            new_key = "model." + k
        else:
            new_key = "model." + k.replace('model', 'language_model')
        new_state_dict[new_key] = v
    
    del state_dict

    if rank == 0:
        print("Injecting weights into pipeline.mllm...")
    
    m, u = pipeline.mllm.load_state_dict(new_state_dict, strict=False)
    
    if rank == 0:
        print(f"Load Results - Missing keys: {len(m)}, Unexpected keys: {len(u)}")
        
        if len(m) > 0:
            real_missing = [k for k in m if "layers" in k or "visual" in k]
            if len(real_missing) > 0:
                print(f"WARNING: Still missing {len(real_missing)} critical keys! First few: {real_missing[:3]}")
            else:
                print("SUCCESS: Critical text encoder weights loaded.")

    try:
        from transformers import AutoProcessor
        pipeline.processor = AutoProcessor.from_pretrained(text_encoder_path, trust_remote_code=True)
        if rank == 0:
            print("Processor updated from finetuned checkpoint.")
    except Exception as e:
        if rank == 0:
            print(f"Could not load processor from checkpoint, using default. Error: {e}")


def load_pipeline(args, device: torch.device, weight_dtype: torch.dtype, rank: int = 0) -> OmniGen2Pipeline:
    """Load the OmniGen2 pipeline with optional finetuned weights."""
    from transformers import CLIPProcessor
    
    if rank == 0:
        print("Loading base OmniGen2 pipeline...")
    
    pipeline = OmniGen2Pipeline.from_pretrained(
        args.model_path,
        processor=CLIPProcessor.from_pretrained(
            args.model_path,
            subfolder="processor",
            use_fast=True
        ),
        torch_dtype=weight_dtype,
        trust_remote_code=True,
    )
    
    if args.transformer_path:
        if rank == 0:
            print(f"Loading finetuned transformer from: {args.transformer_path}")
        pipeline.transformer = OmniGen2Transformer2DModel.from_pretrained(
            args.transformer_path,
            torch_dtype=weight_dtype,
        )
    else:
        pipeline.transformer = OmniGen2Transformer2DModel.from_pretrained(
            args.model_path,
            subfolder="transformer",
            torch_dtype=weight_dtype,
        )

    if args.text_encoder_path:
        load_text_encoder_weights(pipeline, args.text_encoder_path, rank)

    if args.transformer_lora_path:
        if rank == 0:
            print(f"Loading LoRA weights from: {args.transformer_lora_path}")
        pipeline.load_lora_weights(args.transformer_lora_path)

    if args.enable_teacache and args.enable_taylorseer:
        if rank == 0:
            print("WARNING: enable_teacache and enable_taylorseer are mutually exclusive.")

    if args.enable_taylorseer:
        pipeline.enable_taylorseer = True
    elif args.enable_teacache:
        pipeline.transformer.enable_teacache = True
        pipeline.transformer.teacache_rel_l1_thresh = args.teacache_rel_l1_thresh

    if args.scheduler == "dpmsolver++":
        from omnigen2.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
        scheduler = DPMSolverMultistepScheduler(
            algorithm_type="dpmsolver++",
            solver_type="midpoint",
            solver_order=2,
            prediction_type="flow_prediction",
        )
        pipeline.scheduler = scheduler

    if args.enable_sequential_cpu_offload:
        pipeline.enable_sequential_cpu_offload()
    elif args.enable_model_cpu_offload:
        pipeline.enable_model_cpu_offload()
    elif args.enable_group_offload:
        apply_group_offloading(pipeline.transformer, onload_device=device, offload_type="block_level", num_blocks_per_group=2, use_stream=True)
        apply_group_offloading(pipeline.mllm, onload_device=device, offload_type="block_level", num_blocks_per_group=2, use_stream=True)
        apply_group_offloading(pipeline.vae, onload_device=device, offload_type="block_level", num_blocks_per_group=2, use_stream=True)
    else:
        pipeline = pipeline.to(device)

    return pipeline


def process_image_size(image, max_size=1024, min_size=512, stride=16):
    """
    Process image size to be compatible with the model.
    
    Args:
        image: PIL Image
        max_size: Maximum dimension size
        min_size: Minimum dimension size  
        stride: Size must be divisible by stride
        
    Returns:
        Tuple of (width, height)
    """
    def _make_divisible(value, stride):
        return max(stride, int(round(value / stride) * stride))
    
    def _apply_scale(width, height, scale):
        new_width = round(width * scale)
        new_height = round(height * scale)
        new_width = _make_divisible(new_width, stride)
        new_height = _make_divisible(new_height, stride)
        return new_width, new_height
    
    w, h = image.size
    scale = min(max_size / max(w, h), 1.0)
    scale = max(scale, min_size / min(w, h))
    w, h = _apply_scale(w, h, scale)
    
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        w, h = _apply_scale(w, h, scale)
    
    return w, h


def edit_image_with_omnigen2(
    pipe,
    image,
    instruction,
    args,
    device,
    height=None,
    width=None,
    seed=42,
):
    """
    Edit an image using OmniGen2 pipeline.
    
    Args:
        pipe: OmniGen2Pipeline
        image: Input PIL image
        instruction: Text instruction for editing
        args: Command line arguments
        device: torch device
        height: Output height (if None, calculated from input)
        width: Output width (if None, calculated from input)
        seed: Random seed
        
    Returns:
        PIL.Image: Edited image
    """
    # Calculate output size if not provided
    if height is None or width is None:
        width, height = process_image_size(
            image, 
            max_size=args.max_image_size, 
            min_size=args.min_image_size
        )
    
    # Resize input image to match output size
    input_image = image.resize((width, height), Image.LANCZOS)
    
    # Format prompt for image editing
    # OmniGen2 uses <img> placeholder for input images
    prompt = f"<img>\n{instruction}"
    
    # Set seed
    generator = torch.Generator(device=device).manual_seed(seed)
    
    # Generate edited image
    result = pipe(
        prompt=prompt,
        input_images=[input_image],
        height=height,
        width=width,
        num_inference_steps=args.num_inference_steps,
        max_sequence_length=1024,
        text_guidance_scale=args.cfg_text_scale,
        image_guidance_scale=args.cfg_img_scale,
        cfg_range=(args.cfg_range_start, args.cfg_range_end),
        negative_prompt=args.negative_prompt,
        num_images_per_prompt=1,
        generator=generator,
        output_type="pil",
    )
    
    return result.images[0]


def process_dataset(
    pipe,
    args,
    device,
    shard_id=0,
    total_shards=1,
):
    """
    Process images from the GEdit-Bench dataset.
    
    Args:
        pipe: OmniGen2Pipeline
        args: Command line arguments
        device: torch device
        shard_id: Current shard ID for distributed processing
        total_shards: Total number of shards
    """
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset from HuggingFace Hub
    print(f"[Rank {shard_id}] Loading GEdit-Bench dataset...")
    dataset = load_dataset("stepfun-ai/GEdit-Bench")['train']
    
    # Distribute samples across shards
    idx_list = list(range(len(dataset)))
    idx_list = idx_list[shard_id::total_shards]
    
    print(f"[Rank {shard_id}] Processing {len(idx_list)} samples...")
    
    for data_idx in tqdm(idx_list, desc=f"Rank {shard_id}", disable=(shard_id != 0)):
        data = dataset[data_idx]
        
        task_type = data['task_type']
        key = data['key']
        instruction_language = data['instruction_language']
        instruction = data['instruction']
        input_image = data['input_image']
        
        # Create output directories
        save_dir = f"{args.output_dir}/fullset/{task_type}/{instruction_language}"
        os.makedirs(save_dir, exist_ok=True)
        
        save_path_source = f"{save_dir}/{key}_SRCIMG.png"
        save_path_edited = f"{save_dir}/{key}.png"
        
        # Skip if already processed
        if os.path.exists(save_path_source) and os.path.exists(save_path_edited):
            print(f"[Rank {shard_id}] Sample {key} already generated, skipping...")
            continue
        
        try:
            # Edit image
            edited_image = edit_image_with_omnigen2(
                pipe=pipe,
                image=input_image,
                instruction=instruction,
                args=args,
                device=device,
                seed=args.seed,
            )
            
            # Save images
            input_image.save(save_path_source)
            edited_image.save(save_path_edited)
            
            print(f"[Rank {shard_id}] Saved: {save_path_edited}")
            
        except Exception as e:
            print(f"[Rank {shard_id}] Error processing {key}: {e}")
            import traceback
            traceback.print_exc()
            continue


def main():
    parser = argparse.ArgumentParser(description="GEdit Benchmark Evaluation for OmniGen2")
    
    # Model configuration
    parser.add_argument("--model_path", type=str, default='OmniGen2/OmniGen2',
                        help="Path to OmniGen2 model")
    parser.add_argument("--transformer_path", type=str, default=None,
                        help="Path to transformer checkpoint.")
    parser.add_argument("--text_encoder_path", type=str, default=None,
                        help="Path to finetuned text encoder (mllm) checkpoint.")
    parser.add_argument("--transformer_lora_path", type=str, default=None,
                        help="Path to transformer LoRA checkpoint.")
    
    # Output configuration
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save output images")
    
    # Generation parameters
    parser.add_argument("--cfg_text_scale", type=float, default=5.0,
                        help="Text guidance scale")
    parser.add_argument("--cfg_img_scale", type=float, default=2.0,
                        help="Image guidance scale")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                        help="Number of denoising steps")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--cfg_range_start", type=float, default=0.0,
                        help="Start of the CFG range.")
    parser.add_argument("--cfg_range_end", type=float, default=1.0,
                        help="End of the CFG range.")
    parser.add_argument("--negative_prompt", type=str,
                        default="(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar",
                        help="Negative prompt for generation.")
    
    # Image size parameters
    parser.add_argument("--max_image_size", type=int, default=1024,
                        help="Maximum image dimension")
    parser.add_argument("--min_image_size", type=int, default=512,
                        help="Minimum image dimension")
    
    # Precision
    parser.add_argument("--dtype", type=str, default="bf16",
                        choices=["fp32", "fp16", "bf16"],
                        help="Data type for inference")
    
    # Scheduler
    parser.add_argument("--scheduler", type=str, default="euler",
                        choices=["euler", "dpmsolver++"],
                        help="Scheduler to use.")
    
    # Offloading options
    parser.add_argument("--enable_model_cpu_offload", action="store_true",
                        help="Enable model CPU offload.")
    parser.add_argument("--enable_sequential_cpu_offload", action="store_true",
                        help="Enable sequential CPU offload.")
    parser.add_argument("--enable_group_offload", action="store_true",
                        help="Enable group offload.")
    
    # Caching options
    parser.add_argument("--enable_teacache", action="store_true",
                        help="Enable teacache to speed up inference.")
    parser.add_argument("--teacache_rel_l1_thresh", type=float, default=0.05,
                        help="Relative L1 threshold for teacache.")
    parser.add_argument("--enable_taylorseer", action="store_true",
                        help="Enable TaylorSeer Caching.")
    
    args = parser.parse_args()
    
    # Setup distributed environment
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    # Set seeds
    set_seeds(args.seed + rank)
    
    # Select dtype
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    weight_dtype = dtype_map[args.dtype]
    
    # Load pipeline
    if rank == 0:
        print(f"=" * 60)
        print(f"Configuration:")
        print(f"  Model path: {args.model_path}")
        print(f"  Transformer path: {args.transformer_path}")
        print(f"  Text encoder path: {args.text_encoder_path}")
        print(f"  LoRA path: {args.transformer_lora_path}")
        print(f"  Output dir: {args.output_dir}")
        print(f"  Base seed: {args.seed}")
        print(f"=" * 60)
    
    print(f"[Rank {rank}] Loading OmniGen2 pipeline from {args.model_path}...")
    pipe = load_pipeline(args, device, weight_dtype, rank)
    
    # Synchronize after model loading
    if world_size > 1:
        dist.barrier()
        if rank == 0:
            print("All GPUs ready, starting generation...")
    
    # Process dataset
    process_dataset(
        pipe=pipe,
        args=args,
        device=device,
        shard_id=rank,
        total_shards=world_size,
    )
    
    print(f"[Rank {rank}] Completed all tasks!")
    
    # Cleanup with timeout handling
    # if world_size > 1:
    #     try:
    #         dist.barrier(timeout=datetime.timedelta(hours=2))
    #     except Exception as e:
    #         print(f"[Rank {rank}] Barrier timeout/error (this is OK, all images saved): {e}")
    #     finally:
    #         cleanup_distributed()
    
    print(f"[Rank {rank}] Done!")


if __name__ == "__main__":
    main()