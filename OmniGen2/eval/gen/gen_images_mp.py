# geneval_omnigen2.py
import dotenv
dotenv.load_dotenv(override=True)

import os
import json
import argparse
import random
import datetime
import numpy as np
from typing import List

from PIL import Image
import torch

if not torch.cuda.is_available():
    import torch_npu
    from torch_npu.contrib import transfer_to_npu

import torch.distributed as dist
from accelerate import Accelerator
from diffusers.hooks import apply_group_offloading

from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel


def set_seed(seed=0):
    """Set random seed for reproducibility."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="GenEval evaluation for OmniGen2.")
    parser.add_argument(
        "--model_path",
        type=str,
        default='OmniGen2/OmniGen2',
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "--transformer_path",
        type=str,
        default=None,
        help="Path to transformer checkpoint.",
    )
    parser.add_argument(
        "--transformer_lora_path",
        type=str,
        default=None,
        help="Path to transformer LoRA checkpoint.",
    )
    parser.add_argument(
        "--metadata_file",
        type=str,
        default="eval/gen/geneval/prompts/evaluation_metadata.jsonl",
        help="JSONL file containing evaluation metadata.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the generated images.",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=4,
        help="Number of images to generate per prompt.",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="euler",
        choices=["euler", "dpmsolver++"],
        help="Scheduler to use.",
    )
    parser.add_argument(
        "--num_inference_step",
        type=int,
        default=50,
        help="Number of inference steps."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed for generation. Each image will use seed, seed+1, seed+2, ..."
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Output image height."
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Output image width."
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default='bf16',
        choices=['fp32', 'fp16', 'bf16'],
        help="Data type for model weights."
    )
    parser.add_argument(
        "--text_guidance_scale",
        type=float,
        default=4.0,
        help="Text guidance scale."
    )
    parser.add_argument(
        "--image_guidance_scale",
        type=float,
        default=2.0,
        help="Image guidance scale."
    )
    parser.add_argument(
        "--cfg_range_start",
        type=float,
        default=0.0,
        help="Start of the CFG range."
    )
    parser.add_argument(
        "--cfg_range_end",
        type=float,
        default=1.0,
        help="End of the CFG range."
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar",
        help="Negative prompt for generation."
    )
    parser.add_argument(
        "--enable_model_cpu_offload",
        action="store_true",
        help="Enable model CPU offload."
    )
    parser.add_argument(
        "--enable_sequential_cpu_offload",
        action="store_true",
        help="Enable sequential CPU offload."
    )
    parser.add_argument(
        "--enable_group_offload",
        action="store_true",
        help="Enable group offload."
    )
    parser.add_argument(
        "--enable_teacache",
        action="store_true",
        help="Enable teacache to speed up inference."
    )
    parser.add_argument(
        "--teacache_rel_l1_thresh",
        type=float,
        default=0.05,
        help="Relative L1 threshold for teacache."
    )
    parser.add_argument(
        "--enable_taylorseer",
        action="store_true",
        help="Enable TaylorSeer Caching."
    )
    parser.add_argument(
        "--use_distributed",
        action="store_true",
        help="Use distributed processing across multiple GPUs."
    )
    return parser.parse_args()


def load_pipeline(args: argparse.Namespace, device: torch.device, weight_dtype: torch.dtype, rank: int = 0) -> OmniGen2Pipeline:
    """Load the OmniGen2 pipeline."""
    from transformers import CLIPProcessor
    
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
            print(f"Transformer weights loaded from {args.transformer_path}")
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

    if args.transformer_lora_path:
        if rank == 0:
            print(f"LoRA weights loaded from {args.transformer_lora_path}")
        pipeline.load_lora_weights(args.transformer_lora_path)

    if args.enable_teacache and args.enable_taylorseer:
        if rank == 0:
            print("WARNING: enable_teacache and enable_taylorseer are mutually exclusive. enable_teacache will be ignored.")

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


def generate_images(
    pipeline: OmniGen2Pipeline,
    prompt: str,
    args: argparse.Namespace,
    device: torch.device,
    num_images: int = 1,
    base_seed: int = 0,
) -> List[Image.Image]:
    """Generate images using the OmniGen2 pipeline.
    
    Each image uses a distinct seed: base_seed, base_seed+1, base_seed+2, ...
    This guarantees that the images generated from the same prompt differ.
    """
    all_images = []
    
    for i in range(num_images):
        seed = base_seed + i
        generator = torch.Generator(device=device).manual_seed(seed)
        
        results = pipeline(
            prompt=prompt,
            input_images=None,
            width=args.width,
            height=args.height,
            num_inference_steps=args.num_inference_step,
            max_sequence_length=1024,
            text_guidance_scale=args.text_guidance_scale,
            image_guidance_scale=args.image_guidance_scale,
            cfg_range=(args.cfg_range_start, args.cfg_range_end),
            negative_prompt=args.negative_prompt,
            num_images_per_prompt=1,  # Generate one image per call so the seed is effective.
            generator=generator,
            output_type="pil",
        )
        all_images.extend(results.images)
    
    return all_images


def setup_distributed():
    """Setup distributed processing."""
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def main():
    args = parse_args()
    
    # Setup distributed if needed
    if args.use_distributed:
        setup_distributed()
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{rank}")
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    if rank == 0:
        print(f"Output images will be saved in {args.output_dir}")
        print(f"=" * 60)
        print(f"Configuration:")
        print(f"  Model path: {args.model_path}")
        print(f"  Transformer path: {args.transformer_path}")
        print(f"  LoRA path: {args.transformer_lora_path}")
        print(f"  Base seed: {args.seed}")
        print(f"  Num images per prompt: {args.num_images}")
        print(f"  Seeds will be: {args.seed}, {args.seed+1}, ..., {args.seed+args.num_images-1}")
        print(f"=" * 60)
    
    # Set weight dtype
    weight_dtype = torch.float32
    if args.dtype == 'fp16':
        weight_dtype = torch.float16
    elif args.dtype == 'bf16':
        weight_dtype = torch.bfloat16
    
    # Load pipeline
    if rank == 0:
        print("Loading OmniGen2 pipeline...")
    pipeline = load_pipeline(args, device, weight_dtype, rank)
    
    # Synchronize after model loading so every GPU is ready.
    if args.use_distributed:
        dist.barrier()
        if rank == 0:
            print("All GPUs ready, starting generation...")
    
    # Load metadata
    with open(args.metadata_file, "r", encoding="utf-8") as fp:
        metadatas = [json.loads(line) for line in fp]
    total_metadatas = len(metadatas)
    
    if rank == 0:
        print(f"Total prompts: {total_metadatas}")
    
    # Distribute prompts across GPUs
    prompts_per_gpu = (total_metadatas + world_size - 1) // world_size
    start = rank * prompts_per_gpu
    end = min(start + prompts_per_gpu, total_metadatas)
    
    print(f"GPU {rank}: Processing {end - start} prompts (indices {start} to {end - 1})")
    
    # Process each prompt
    for idx in range(start, end):
        metadata = metadatas[idx]
        prompt = metadata['prompt']
        
        # Create output directory for this prompt
        outpath = os.path.join(args.output_dir, f"{idx:0>5}")
        os.makedirs(outpath, exist_ok=True)
        
        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)
        
        # Check if already generated
        all_exist = True
        for img_idx in range(args.num_images):
            if not os.path.exists(os.path.join(sample_path, f"{img_idx:05}.png")):
                all_exist = False
                break
        
        if all_exist:
            print(f"GPU {rank}: Skipping prompt {idx} (already generated): '{prompt}'")
            continue
        
        print(f"GPU {rank}: Processing prompt {idx - start + 1}/{end - start} (global idx {idx}): '{prompt}'")
        
        # Save metadata
        with open(os.path.join(outpath, "metadata.jsonl"), "w", encoding="utf-8") as fp:
            json.dump(metadata, fp)
        
        # Generate images - one distinct seed per image
        try:
            image_list = generate_images(
                pipeline=pipeline,
                prompt=prompt,
                args=args,
                device=device,
                num_images=args.num_images,
                base_seed=args.seed,
            )
        except Exception as e:
            print(f"GPU {rank}: Error generating images for prompt {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Save generated images
        for sample_idx, sample in enumerate(image_list):
            # Optionally crop to bounding box (remove if not needed)
            bbox = sample.getbbox()
            if bbox:
                sample = sample.crop(bbox)
            sample.save(os.path.join(sample_path, f"{sample_idx:05}.png"))
        
        print(f"GPU {rank}: Saved {len(image_list)} images for prompt {idx} (seeds: {args.seed} to {args.seed + args.num_images - 1})")
    
    print(f"GPU {rank}: Completed all tasks")
    
    # Increase the barrier timeout and handle exceptions gracefully.
    if args.use_distributed:
        try:
            dist.barrier(timeout=datetime.timedelta(hours=2))
        except Exception as e:
            print(f"GPU {rank}: Barrier timeout/error (this is OK, all images saved): {e}")
        finally:
            try:
                dist.destroy_process_group()
            except:
                pass


if __name__ == "__main__":
    main()