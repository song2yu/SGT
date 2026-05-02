# Copyright 2025 OmniGen2 Team
# Text-to-image demo for OmniGen2 / SGT-Gen2.
#
# This script mirrors the style of ``scripts/infer_edit.py``:
#   * It resolves the project root so ``omnigen2.*`` imports work from any CWD.
#   * ``--checkpoint_dir`` mirrors the ``CHECKPOINT_DIR`` pattern used by
#     ``scripts/infra/infer_gedit.sh``: given a directory that contains
#     ``transformer/transformer`` and ``transformer/text_encoder`` sub-folders,
#     both finetuned components are loaded automatically.
#
# Generation logic follows ``scripts/infra/inference.py`` /
# ``scripts/infra/inference_chat.py``: we call ``OmniGen2Pipeline`` directly
# with ``input_images=None``, which produces a pure text-to-image sample.
#
# Default example reproduces ``scripts/infra/example_t2i.sh``:
#   instruction = "The sun rises slightly, the dew on the rose petals ..."
#   resolution  = 1024 x 1024
#   steps       = 50
#

import argparse
import glob
import os
import random
import sys

import numpy as np
import torch
from safetensors.torch import load_file

# This file lives at <project_root>/scripts/infer_text2image.py; make the
# project root importable so that ``omnigen2.*`` modules can be found
# regardless of CWD.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if not torch.cuda.is_available():
    # Allow running on Ascend NPU with the same code path.
    import torch_npu  # noqa: F401
    from torch_npu.contrib import transfer_to_npu  # noqa: F401

from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel
from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline


DEFAULT_NEGATIVE_PROMPT = (
    "(((deformed))), blurry, over saturation, bad anatomy, disfigured, "
    "poorly drawn face, mutation, mutated, (extra_limb), (ugly), "
    "(poorly drawn hands), fused fingers, messy drawing, broken legs censor, "
    "censored, censor_bar"
)

DEFAULT_INSTRUCTION = (
    "The sun rises slightly, the dew on the rose petals in the garden is "
    "clear, a crystal ladybug is crawling to the dew, the background is the "
    "early morning garden, macro lens."
)


def set_seeds(seed):
    """Set random seeds for reproducibility."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Finetuned text-encoder loading (same remap as infer_edit.py)
# ---------------------------------------------------------------------------

def load_text_encoder_weights(pipeline, text_encoder_path):
    """Load finetuned text encoder (mllm) weights with key remapping.

    Mirrors ``infer_edit.load_text_encoder_weights`` so the SGT-Gen2
    checkpoints can be consumed by this script as well.
    """
    print(f"Loading finetuned text encoder weights from: {text_encoder_path}")
    files = sorted(glob.glob(os.path.join(text_encoder_path, "*.safetensors")))
    if not files:
        files = sorted(glob.glob(os.path.join(text_encoder_path, "*.bin")))
    if not files:
        raise FileNotFoundError(f"No weights found in {text_encoder_path}")

    state_dict = {}
    for f in files:
        if f.endswith(".safetensors"):
            state_dict.update(load_file(f))
        else:
            state_dict.update(torch.load(f, map_location="cpu"))

    new_state_dict = {}
    for k, v in state_dict.items():
        if not k.startswith("model."):
            new_key = "model." + k
        else:
            new_key = "model." + k.replace("model", "language_model")
        new_state_dict[new_key] = v

    missing, unexpected = pipeline.mllm.load_state_dict(
        new_state_dict, strict=False
    )
    print(
        f"Text encoder load: missing={len(missing)}, unexpected={len(unexpected)}"
    )

    # Try to also pick up the processor shipped with the checkpoint.
    try:
        from transformers import AutoProcessor

        pipeline.processor = AutoProcessor.from_pretrained(
            text_encoder_path, trust_remote_code=True
        )
        print("Processor updated from finetuned checkpoint.")
    except Exception as e:  # pragma: no cover - best-effort fallback
        print(f"Could not load processor from checkpoint, using default. Error: {e}")


# ---------------------------------------------------------------------------
# Pipeline loading
# ---------------------------------------------------------------------------

def load_pipeline(args, device, weight_dtype):
    """Create an ``OmniGen2Pipeline`` with optional finetuned weights.

    If ``args.checkpoint_dir`` is set, the finetuned transformer and text
    encoder are loaded from::

        <checkpoint_dir>/transformer/transformer
        <checkpoint_dir>/transformer/text_encoder

    mirroring the paths used in ``scripts/infra/infer_gedit.sh``.
    """
    from transformers import CLIPProcessor

    # Derive component paths from a single checkpoint directory (sh-style).
    transformer_path = None
    text_encoder_path = None
    if args.checkpoint_dir:
        ckpt_dir = os.path.abspath(args.checkpoint_dir)
        transformer_path = os.path.join(ckpt_dir, "transformer", "transformer")
        text_encoder_path = os.path.join(ckpt_dir, "transformer", "text_encoder")
        if not os.path.isdir(transformer_path):
            raise FileNotFoundError(
                f"Expected finetuned transformer directory not found: {transformer_path}"
            )
        if not os.path.isdir(text_encoder_path):
            raise FileNotFoundError(
                f"Expected finetuned text encoder directory not found: {text_encoder_path}"
            )

    print(f"Loading base OmniGen2 pipeline from: {args.model_path}")
    pipeline = OmniGen2Pipeline.from_pretrained(
        args.model_path,
        processor=CLIPProcessor.from_pretrained(
            args.model_path, subfolder="processor", use_fast=True
        ),
        torch_dtype=weight_dtype,
        trust_remote_code=True,
    )

    if transformer_path:
        print(f"Loading finetuned transformer from: {transformer_path}")
        pipeline.transformer = OmniGen2Transformer2DModel.from_pretrained(
            transformer_path, torch_dtype=weight_dtype
        )
    else:
        pipeline.transformer = OmniGen2Transformer2DModel.from_pretrained(
            args.model_path, subfolder="transformer", torch_dtype=weight_dtype
        )

    if text_encoder_path:
        load_text_encoder_weights(pipeline, text_encoder_path)

    if args.transformer_lora_path:
        print(f"Loading LoRA weights from: {args.transformer_lora_path}")
        pipeline.load_lora_weights(args.transformer_lora_path)

    # Optional scheduler override.
    if args.scheduler == "dpmsolver++":
        from omnigen2.schedulers.scheduling_dpmsolver_multistep import (
            DPMSolverMultistepScheduler,
        )

        pipeline.scheduler = DPMSolverMultistepScheduler(
            algorithm_type="dpmsolver++",
            solver_type="midpoint",
            solver_order=2,
            prediction_type="flow_prediction",
        )

    if args.enable_model_cpu_offload:
        pipeline.enable_model_cpu_offload()
    elif args.enable_sequential_cpu_offload:
        pipeline.enable_sequential_cpu_offload()
    else:
        pipeline = pipeline.to(device)

    return pipeline


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_image(pipeline, instruction, args, device):
    """Run a single text-to-image generation and return PIL images."""
    generator = torch.Generator(device=device).manual_seed(args.seed)

    result = pipeline(
        prompt=instruction,
        input_images=None,  # text-to-image: no reference images.
        width=args.width,
        height=args.height,
        num_inference_steps=args.num_inference_steps,
        max_sequence_length=1024,
        text_guidance_scale=args.cfg_text_scale,
        image_guidance_scale=args.cfg_img_scale,
        cfg_range=(args.cfg_range_start, args.cfg_range_end),
        negative_prompt=args.negative_prompt,
        num_images_per_prompt=args.num_images_per_prompt,
        generator=generator,
        output_type="pil",
    )
    return result.images


# ---------------------------------------------------------------------------
# Argument parsing / main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Text-to-image demo for OmniGen2 / SGT-Gen2. Reproduces "
            "scripts/infra/example_t2i.sh by default."
        )
    )

    # Prompt / output
    parser.add_argument(
        "--instruction",
        type=str,
        default=DEFAULT_INSTRUCTION,
        help="Text prompt for generation.",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=DEFAULT_NEGATIVE_PROMPT,
        help="Negative prompt for generation.",
    )
    parser.add_argument(
        "--output_image_path",
        type=str,
        default="outputs/output_t2i.png",
        help="Where to save the generated image.",
    )
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=1,
        help="Number of images to sample per prompt.",
    )

    # Model paths
    parser.add_argument(
        "--model_path",
        type=str,
        default="pretrained_models/OmniGen2",
        help="HuggingFace repo id or local path of the base OmniGen2 model.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="pretrained_models/SGT-Gen2",
        help=(
            "Optional path to a finetuned checkpoint directory. The script "
            "expects `<checkpoint_dir>/transformer/transformer` and "
            "`<checkpoint_dir>/transformer/text_encoder` to exist, mirroring "
            "the layout used by scripts/infer_edit.py. Pass an empty string "
            "to skip and use the base model only."
        ),
    )
    parser.add_argument(
        "--transformer_lora_path",
        type=str,
        default=None,
        help="Optional path to LoRA weights for the transformer.",
    )

    # Generation parameters
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument(
        "--cfg_text_scale",
        type=float,
        default=4.0,
        help="Text classifier-free guidance scale (example_t2i.sh uses 4.0).",
    )
    parser.add_argument(
        "--cfg_img_scale",
        type=float,
        default=1.0,
        help=(
            "Image guidance scale. For pure text-to-image this is effectively "
            "ignored (set to 1.0 by the pipeline when no input images)."
        ),
    )
    parser.add_argument("--cfg_range_start", type=float, default=0.0)
    parser.add_argument("--cfg_range_end", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--scheduler",
        type=str,
        default="euler",
        choices=["euler", "dpmsolver++"],
        help="Sampling scheduler to use.",
    )

    # Size parameters
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)

    # Runtime
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["fp32", "fp16", "bf16"],
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on. Defaults to cuda if available, otherwise cpu.",
    )
    parser.add_argument("--enable_model_cpu_offload", action="store_true")
    parser.add_argument("--enable_sequential_cpu_offload", action="store_true")

    return parser.parse_args()


def save_images(images, output_path):
    """Save one or more PIL images to ``output_path``.

    When multiple images are produced, they are written as
    ``<stem>_{i}<ext>``; the first image is always written to the exact path
    the user requested.
    """
    os.makedirs(
        os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True
    )

    if len(images) == 1:
        images[0].save(output_path)
        print(f"Saved image to: {output_path}")
        return

    stem, ext = os.path.splitext(output_path)
    for i, img in enumerate(images):
        path_i = output_path if i == 0 else f"{stem}_{i}{ext}"
        img.save(path_i)
        print(f"Saved image to: {path_i}")


def main():
    args = parse_args()

    set_seeds(args.seed)

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    weight_dtype = dtype_map[args.dtype]

    print("=" * 60)
    print("OmniGen2 text-to-image demo")
    print(f"  instruction   : {args.instruction}")
    print(f"  output_path   : {args.output_image_path}")
    print(f"  model_path    : {args.model_path}")
    print(f"  checkpoint_dir: {args.checkpoint_dir}")
    print(f"  device/dtype  : {device} / {args.dtype}")
    print(f"  seed/steps    : {args.seed} / {args.num_inference_steps}")
    print(f"  size (HxW)    : {args.height} x {args.width}")
    print(f"  cfg text/img  : {args.cfg_text_scale} / {args.cfg_img_scale}")
    print(f"  num_images    : {args.num_images_per_prompt}")
    print("=" * 60)

    pipeline = load_pipeline(args, device, weight_dtype)

    images = generate_image(pipeline, args.instruction, args, device)

    save_images(images, args.output_image_path)


if __name__ == "__main__":
    main()
