# Copyright 2025 OmniGen2 Team
# Single-image editing demo for OmniGen2.
#
# This script lives under ``scripts/`` and expects to be run from the
# project root (or any CWD) -- it resolves the project root itself.
#
# The ``--checkpoint_dir`` argument mirrors the ``CHECKPOINT_DIR`` variable in
# ``scripts/infra/infer_gedit.sh``: given a directory that contains
# ``transformer/transformer`` and ``transformer/text_encoder`` sub-folders,
# both finetuned components are loaded automatically.
#
# Example:
#   # Base model only
#   python scripts/infer_edit.py \
#       --input_image example_images/1.png \
#       --instruction "Turn the background into a snowy mountain scene." \
#       --output_path outputs/edit_demo.png
#
#   # With a finetuned checkpoint directory
#   python scripts/infer_edit.py \
#       --input_image example_images/1.png \
#       --instruction "Turn the background into a snowy mountain scene." \
#       --output_path outputs/edit_demo.png \
#       --checkpoint_dir experiments/<run>/checkpoint-<step>/

import argparse
import glob
import os
import random
import sys

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

# This file lives at <project_root>/scripts/infer_edit.py; make the project
# root importable so that `omnigen2.*` modules can be found regardless of CWD.
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


def load_text_encoder_weights(pipeline, text_encoder_path):
    """Load finetuned text encoder (mllm) weights with key remapping.

    Mirrors the logic used in ``eval/gen/gen_gedit.py`` so the same
    checkpoints can be consumed by this demo script.
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

    missing, unexpected = pipeline.mllm.load_state_dict(new_state_dict, strict=False)
    print(
        f"Text encoder load: missing={len(missing)}, unexpected={len(unexpected)}"
    )

    try:
        from transformers import AutoProcessor

        pipeline.processor = AutoProcessor.from_pretrained(
            text_encoder_path, trust_remote_code=True
        )
        print("Processor updated from finetuned checkpoint.")
    except Exception as e:
        print(f"Could not load processor from checkpoint, using default. Error: {e}")


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

    if args.enable_model_cpu_offload:
        pipeline.enable_model_cpu_offload()
    elif args.enable_sequential_cpu_offload:
        pipeline.enable_sequential_cpu_offload()
    else:
        pipeline = pipeline.to(device)

    return pipeline


def compute_output_size(image, max_size=1024, min_size=512, stride=16):
    """Rescale the input image so its dims fit within [min_size, max_size]
    and are divisible by ``stride``.

    Returns the target (width, height).
    """

    def _make_divisible(value, stride_):
        return max(stride_, int(round(value / stride_) * stride_))

    def _apply_scale(w_, h_, scale):
        new_w = _make_divisible(round(w_ * scale), stride)
        new_h = _make_divisible(round(h_ * scale), stride)
        return new_w, new_h

    w, h = image.size
    scale = min(max_size / max(w, h), 1.0)
    scale = max(scale, min_size / min(w, h))
    w, h = _apply_scale(w, h, scale)

    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        w, h = _apply_scale(w, h, scale)

    return w, h


def edit_image(pipeline, image, instruction, args, device):
    """Run the editing pipeline on a single image + instruction."""
    if args.height and args.width:
        width, height = args.width, args.height
    else:
        width, height = compute_output_size(
            image, max_size=args.max_image_size, min_size=args.min_image_size
        )

    input_image = image.resize((width, height), Image.LANCZOS)

    # OmniGen2 expects an <img> placeholder in the prompt for every input image.
    prompt = f"<img>\n{instruction}"

    generator = torch.Generator(device=device).manual_seed(args.seed)

    result = pipeline(
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
    return result.images[0], input_image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Single-image editing demo for OmniGen2."
    )

    # Inputs
    parser.add_argument(
        "--input_image",
        type=str,
        default="../assets/图片3.png",
        help="Path to the input image to edit.",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="Turn the background into a snowy mountain scene.",
        help="Natural-language editing instruction.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="outputs/edit_demo.png",
        help="Where to save the edited image.",
    )
    parser.add_argument(
        "--save_source",
        action="store_true",
        help="Also save the (resized) source image next to the output.",
    )

    # Model paths
    parser.add_argument(
        "--model_path",
        type=str,
        default="pretrained_models/OmniGen2",
        help="HuggingFace repo id or local path of the base OmniGen2 model.",
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default='pretrained_models/SGT-Gen2',
        help=(
            "Optional path to a finetuned checkpoint directory. The script "
            "expects `<checkpoint_dir>/transformer/transformer` and "
            "`<checkpoint_dir>/transformer/text_encoder` to exist, mirroring "
            "the layout used by scripts/infra/infer_gedit.sh."
        ),
    )
    parser.add_argument(
        "--transformer_lora_path", type=str, default=None,
        help="Optional path to LoRA weights for the transformer."
    )

    # Generation parameters
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--cfg_text_scale", type=float, default=5.0)
    parser.add_argument("--cfg_img_scale", type=float, default=2.0)
    parser.add_argument("--cfg_range_start", type=float, default=0.0)
    parser.add_argument("--cfg_range_end", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument(
        "--negative_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT
    )

    # Size parameters
    parser.add_argument("--max_image_size", type=int, default=1024)
    parser.add_argument("--min_image_size", type=int, default=512)
    parser.add_argument(
        "--height", type=int, default=None,
        help="Force output height (must be a multiple of 16)."
    )
    parser.add_argument(
        "--width", type=int, default=None,
        help="Force output width (must be a multiple of 16)."
    )

    # Runtime
    parser.add_argument(
        "--dtype", type=str, default="bf16",
        choices=["fp32", "fp16", "bf16"],
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to run on. Defaults to cuda if available, otherwise cpu."
    )
    parser.add_argument("--enable_model_cpu_offload", action="store_true")
    parser.add_argument("--enable_sequential_cpu_offload", action="store_true")

    return parser.parse_args()


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
    print("OmniGen2 single-image edit demo")
    print(f"  input_image : {args.input_image}")
    print(f"  instruction : {args.instruction}")
    print(f"  output_path : {args.output_path}")
    print(f"  model_path  : {args.model_path}")
    print(f"  device/dtype: {device} / {args.dtype}")
    print(f"  seed/steps  : {args.seed} / {args.num_inference_steps}")
    print("=" * 60)

    if not os.path.exists(args.input_image):
        raise FileNotFoundError(f"Input image not found: {args.input_image}")
    image = Image.open(args.input_image).convert("RGB")

    pipeline = load_pipeline(args, device, weight_dtype)

    edited, source = edit_image(pipeline, image, args.instruction, args, device)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)) or ".", exist_ok=True)
    edited.save(args.output_path)
    print(f"Saved edited image to: {args.output_path}")

    if args.save_source:
        src_path = os.path.splitext(args.output_path)[0] + "_source.png"
        source.save(src_path)
        print(f"Saved resized source image to: {src_path}")


if __name__ == "__main__":
    main()
