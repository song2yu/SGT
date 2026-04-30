import dotenv

dotenv.load_dotenv(override=True)

import argparse
import os
from typing import List, Tuple

from PIL import Image, ImageOps

import torch
if not torch.cuda.is_available():
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
# from torchvision.transforms.functional import to_pil_image, to_tensor

from accelerate import Accelerator
from diffusers.hooks import apply_group_offloading

from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel
from PIL import Image
import numpy as np
from typing import Union
def to_tensor(pic: Union[Image.Image, np.ndarray]) -> torch.Tensor:
    """
    Converts a PIL Image or a NumPy array to a PyTorch Tensor.
    This function is a replacement for torchvision.transforms.functional.to_tensor.

    Args:
        pic (PIL.Image.Image or np.ndarray): Image to be converted.
            - If PIL Image, it is converted to a NumPy array.
            - If NumPy array, shape should be (H, W, C) in [0, 255] range.
            - Grayscale images (H, W) are also handled.

    Returns:
        torch.Tensor: The converted image tensor. Shape: (C, H, W).
                      Data is a float32 tensor in the [0.0, 1.0] range.
    """
    # 1. Convert PIL Image to NumPy array if necessary
    if isinstance(pic, Image.Image):
        img = np.array(pic)
    elif isinstance(pic, np.ndarray):
        img = pic
    else:
        raise TypeError(f"Input type {type(pic)} is not a PIL Image or NumPy array.")

    # Handle grayscale images by adding a channel dimension
    if img.ndim == 2:
        # Shape: (H, W) -> (H, W, 1)
        img = np.expand_dims(img, axis=2)

    # 2. Transpose dimensions from (H, W, C) to (C, H, W)
    # This is the most critical step for PyTorch compatibility
    img = img.transpose((2, 0, 1))
    
    # 3. Convert NumPy array to PyTorch Tensor
    # np.ascontiguousarray is a good practice before from_numpy
    tensor = torch.from_numpy(np.ascontiguousarray(img))
    
    # 4. Scale from [0, 255] to [0.0, 1.0] and convert to float32
    # The original to_tensor always returns a float32 tensor
    return tensor.to(torch.float32).div(255.0)
    
def to_pil_image(tensor: Union[torch.Tensor, np.ndarray]) -> Image.Image:
    """
    Converts a PyTorch Tensor or a NumPy array to a PIL Image.
    This function is a replacement for torchvision.transforms.functional.to_pil_image.

    Args:
        tensor (torch.Tensor or np.ndarray): Image to be converted.
            - If torch.Tensor, shape should be (C, H, W) or (H, W).
            - If np.ndarray, shape should be (H, W, C) or (H, W).
            The tensor can be on any device (e.g., NPU, CPU), it will be moved to CPU.
            The tensor data type can be float (values in [0, 1]) or uint8 (values in [0, 255]).

    Returns:
        PIL.Image.Image: The converted image.
    """
    # 1. Ensure input is a torch.Tensor
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
        # If numpy array is HWC, we will handle it later
    elif not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Input type {type(tensor)} is not a torch.Tensor or np.ndarray.")

    # 2. Move tensor to CPU if it's not already there
    if tensor.device.type != 'cpu':
        tensor = tensor.cpu()
        
    # 3. Handle data type conversion (scaling for floats)
    #    to_pil_image expects data in the [0, 255] range as uint8
    if tensor.is_floating_point():
        # Scale from [0.0, 1.0] to [0, 255]
        tensor = tensor.mul(255).clamp(0, 255)

    # Convert to uint8 data type
    tensor = tensor.to(torch.uint8)

    # 4. Convert tensor to NumPy array
    numpy_array = tensor.numpy()

    # 5. Handle dimension order
    # PyTorch tensors are typically (C, H, W) or (H, W) for grayscale
    # PIL/NumPy expect (H, W, C) for color images or (H, W) for grayscale
    if numpy_array.ndim == 3:
        # If it has 3 dimensions, it must be C, H, W. Transpose to H, W, C.
        numpy_array = numpy_array.transpose(1, 2, 0)
    
    # Handle the case where a (1, H, W) tensor is passed for grayscale
    if numpy_array.shape[-1] == 1:
        numpy_array = numpy_array.squeeze(axis=-1)

    # 6. Create PIL Image from the NumPy array
    image = Image.fromarray(numpy_array)

    return image


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="OmniGen2 image generation script.")
    parser.add_argument(
        "--model_path",
        type=str,
        default='OmniGen2/OmniGen2', # OmniGen2/OmniGen2
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
        default=None, #  'experiments/ft_lora_bs_2_ori/transformer_lora'
        help="Path to transformer LoRA checkpoint.",
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
        help="Random seed for generation."
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
        "--max_input_image_pixels",
        type=int,
        default=1048576,
        help="Maximum number of pixels for each input image."
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
        default=5.0,
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
        "--instruction",
        type=str,
        default="",
        help="Text prompt for generation."
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar",
        help="Negative prompt for generation."
    )
    parser.add_argument(
        "--input_image_path",
        type=str,
        nargs='+',
        default='outputs/third_bird.jpg',
        help="Path(s) to input image(s)."
    )
    parser.add_argument(
        "--output_image_path",
        type=str,
        default="outputs/blur_outut.png",
        help="Path to save output image."
    )
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=1,
        help="Number of images to generate per prompt."
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
    return parser.parse_args()

def load_pipeline(args: argparse.Namespace, accelerator: Accelerator, weight_dtype: torch.dtype) -> OmniGen2Pipeline:
    # pipeline = OmniGen2Pipeline.from_pretrained(
    #     args.model_path,
    #     torch_dtype=weight_dtype,
    #     trust_remote_code=True,
    # )
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
        print(f"LoRA weights loaded from {args.transformer_lora_path}")
        pipeline.load_lora_weights(args.transformer_lora_path)

    if args.enable_teacache and args.enable_taylorseer:
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
        apply_group_offloading(pipeline.transformer, onload_device=accelerator.device, offload_type="block_level", num_blocks_per_group=2, use_stream=True)
        apply_group_offloading(pipeline.mllm, onload_device=accelerator.device, offload_type="block_level", num_blocks_per_group=2, use_stream=True)
        apply_group_offloading(pipeline.vae, onload_device=accelerator.device, offload_type="block_level", num_blocks_per_group=2, use_stream=True)
    else:
        pipeline = pipeline.to(accelerator.device)

    return pipeline

def preprocess(input_image_path: List[str] = []) -> Tuple[str, str, List[Image.Image]]:
    """Preprocess the input images."""
    # Process input images
    input_images = None

    if input_image_path:
        input_images = []
        if isinstance(input_image_path, str):
            input_image_path = [input_image_path]

        if len(input_image_path) == 1 and os.path.isdir(input_image_path[0]):
            input_images = [Image.open(os.path.join(input_image_path[0], f)).convert("RGB")
                          for f in os.listdir(input_image_path[0])]
        else:
            input_images = [Image.open(path).convert("RGB") for path in input_image_path]

        input_images = [ImageOps.exif_transpose(img) for img in input_images]

    return input_images

def run(args: argparse.Namespace, 
        accelerator: Accelerator, 
        pipeline: OmniGen2Pipeline, 
        instruction: str, 
        negative_prompt: str, 
        input_images: List[Image.Image]) -> Image.Image:
    """Run the image generation pipeline with the given parameters."""
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    results = pipeline(
        prompt=instruction,
        input_images=input_images,
        width=args.width,
        height=args.height,
        num_inference_steps=args.num_inference_step,
        max_sequence_length=1024,
        text_guidance_scale=args.text_guidance_scale,
        image_guidance_scale=args.image_guidance_scale,
        cfg_range=(args.cfg_range_start, args.cfg_range_end),
        negative_prompt=negative_prompt,
        num_images_per_prompt=args.num_images_per_prompt,
        generator=generator,
        output_type="pil",
    )
    return results

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

def main(args: argparse.Namespace, root_dir: str) -> None:
    """Main function to run the image generation process."""
    # Initialize accelerator
    accelerator = Accelerator(mixed_precision=args.dtype if args.dtype != 'fp32' else 'no')

    # Set weight dtype
    weight_dtype = torch.float32
    if args.dtype == 'fp16':
        weight_dtype = torch.float16
    elif args.dtype == 'bf16':
        weight_dtype = torch.bfloat16

    # Load pipeline and process inputs
    pipeline = load_pipeline(args, accelerator, weight_dtype)
    input_images = preprocess(args.input_image_path)

    # Generate and save image
    results = run(args, accelerator, pipeline, args.instruction, args.negative_prompt, input_images)
    os.makedirs(os.path.dirname(args.output_image_path), exist_ok=True)

    if len(results.images) > 1:
        for i, image in enumerate(results.images):
            image_name, ext = os.path.splitext(args.output_image_path)
            image.save(f"{image_name}_{i}{ext}")

    vis_images = [to_tensor(image) * 2 - 1 for image in results.images]
    output_image = create_collage(vis_images)

    output_image.save(args.output_image_path)
    print(f"Image saved to {args.output_image_path}")

if __name__ == "__main__":
    root_dir = os.path.abspath(os.path.join(__file__, os.path.pardir))
    args = parse_args()
    main(args, root_dir)