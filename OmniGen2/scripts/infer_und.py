# Copyright 2025 OmniGen2 Team
# Visual understanding (image chat) demo for OmniGen2 / SGT-Gen2.
#
# This script mirrors the style of ``scripts/infer_edit.py``:
#   * It resolves the project root so ``omnigen2.*`` imports work from any CWD.
#   * ``--checkpoint_dir`` mirrors the ``CHECKPOINT_DIR`` pattern used by
#     ``scripts/infra/infer_gedit.sh``: given a directory that contains
#     ``transformer/text_encoder`` (and optionally ``transformer/transformer``),
#     the fine-tuned MLLM weights are loaded automatically.
#
# Chat flow:
#   Visual understanding runs Qwen2.5-VL's native ``forward`` via
#   ``mllm.generate``. That path REQUIRES the processor to emit
#   ``pixel_values`` + ``image_grid_thw`` alongside the text tokens. The only
#   processor/template combination that guarantees this is Qwen2.5-VL's
#   official chat template applied through ``AutoProcessor`` on a
#   ``[{"type": "image"}, {"type": "text", ...}]`` message. We therefore load
#   ``<model_path>/mllm_processor`` (a full Qwen2.5-VL processor) rather than
#   the CLIP-style ``processor`` sub-folder that OmniGen2's diffusion pipeline
#   uses.
#
# Default example reproduces ``scripts/infra/example_understanding.sh``:
#   instruction = "Please describe this image briefly."
#   image       = example_images/02.jpg
#

import argparse
import glob
import os
import sys

import torch
from PIL import Image, ImageOps
from safetensors.torch import load_file

# This file lives at <project_root>/scripts/infer_und.py; make the project
# root importable so that ``omnigen2.*`` modules can be found regardless of CWD.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if not torch.cuda.is_available():
    # Allow running on Ascend NPU with the same code path.
    import torch_npu  # noqa: F401
    from torch_npu.contrib import transfer_to_npu  # noqa: F401

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


# ---------------------------------------------------------------------------
# Finetuned text-encoder loading (same remap as infer_edit.py)
# ---------------------------------------------------------------------------

def load_text_encoder_weights(mllm, text_encoder_path):
    """Load fine-tuned text encoder (mllm) weights with key remapping.

    Mirrors ``infer_edit.load_text_encoder_weights`` so the SGT-Gen2
    checkpoints can be consumed here as well.
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

    missing, unexpected = mllm.load_state_dict(new_state_dict, strict=False)
    print(
        f"Text encoder load: missing={len(missing)}, unexpected={len(unexpected)}"
    )

    # Warn only if critical vision / language layers are missing after load.
    if missing:
        critical = [k for k in missing if ("layers" in k or "visual" in k)]
        if critical:
            print(
                f"   WARNING: still missing {len(critical)} critical keys "
                f"(e.g. {critical[0]})."
            )


# ---------------------------------------------------------------------------
# Model / processor loading
# ---------------------------------------------------------------------------

def _resolve_mllm_dir(model_path: str) -> str:
    """Return the directory to load the MLLM weights from."""
    mllm_dir = os.path.join(model_path, "mllm")
    if os.path.isdir(mllm_dir):
        return mllm_dir
    # Fall back to the repo root (e.g. when ``model_path`` is a HF repo id).
    return model_path


def _resolve_processor_dir(model_path: str) -> str:
    """Return the directory containing the full Qwen2.5-VL processor.

    OmniGen2 ships two processor folders:
      * ``processor``        -- CLIP-style, used by the diffusion pipeline.
      * ``mllm_processor``   -- full Qwen2.5-VL processor (tokenizer +
                                image processor + chat_template.json).

    For visual understanding we need the latter so that apply_chat_template
    and the image processor together produce ``pixel_values`` +
    ``image_grid_thw`` for the vision branch.
    """
    for name in ("mllm_processor", "processor"):
        p = os.path.join(model_path, name)
        if os.path.isdir(p) and os.path.isfile(
            os.path.join(p, "preprocessor_config.json")
        ):
            return p
    # Fall back to the repo root -- AutoProcessor will try to locate files.
    return model_path


def load_mllm_and_processor(args, device, weight_dtype):
    """Load the Qwen2.5-VL MLLM + its full (image-aware) processor.

    Only the multimodal language model is required for visual understanding;
    no VAE / transformer / scheduler is loaded. When ``--checkpoint_dir`` is
    provided, the fine-tuned SGT text-encoder weights are overlaid on top.
    """
    # --- Processor ---------------------------------------------------------
    processor_dir = _resolve_processor_dir(args.model_path)
    print(f"Loading Qwen2.5-VL processor from: {processor_dir}")
    processor = AutoProcessor.from_pretrained(
        processor_dir, trust_remote_code=True
    )

    # --- MLLM --------------------------------------------------------------
    mllm_dir = _resolve_mllm_dir(args.model_path)
    print(f"Loading base MLLM from: {mllm_dir}")
    mllm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        mllm_dir,
        torch_dtype=weight_dtype,
        trust_remote_code=True,
    )

    # --- Optional: overlay fine-tuned SGT text-encoder weights -------------
    text_encoder_path = None
    if args.checkpoint_dir:
        ckpt_dir = os.path.abspath(args.checkpoint_dir)
        text_encoder_path = os.path.join(ckpt_dir, "transformer", "text_encoder")
        if not os.path.isdir(text_encoder_path):
            raise FileNotFoundError(
                f"Expected finetuned text encoder directory not found: "
                f"{text_encoder_path}"
            )

    if text_encoder_path:
        load_text_encoder_weights(mllm, text_encoder_path)

        # If the finetuned checkpoint ships its own full processor, prefer it
        # -- but only when it is actually a Qwen2.5-VL processor (must include
        # ``preprocessor_config.json`` so it can produce pixel_values).
        if os.path.isfile(os.path.join(text_encoder_path, "preprocessor_config.json")):
            try:
                processor = AutoProcessor.from_pretrained(
                    text_encoder_path, trust_remote_code=True
                )
                print("Processor updated from finetuned checkpoint.")
            except Exception as e:  # pragma: no cover - best-effort fallback
                print(
                    f"Could not load processor from checkpoint, keeping "
                    f"default. Error: {e}"
                )

    mllm.eval()
    mllm.to(device)
    return mllm, processor


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _build_messages(instruction: str, num_images: int):
    """Build a Qwen2.5-VL-style chat message with ``num_images`` images."""
    content = []
    for _ in range(num_images):
        content.append({"type": "image"})
    content.append({"type": "text", "text": instruction})
    return [{"role": "user", "content": content}]


@torch.no_grad()
def understand_image(
    mllm,
    processor,
    images,
    instruction: str,
    device,
    max_new_tokens: int = 256,
    do_sample: bool = False,
) -> str:
    """Run a single image-understanding turn and return the answer text."""
    num_images = len(images) if images else 0
    messages = _build_messages(instruction, num_images=num_images)

    text_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[text_prompt],
        images=images if num_images > 0 else None,
        videos=None,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    generated_ids = mllm.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
    )

    # Trim the prompt tokens to isolate the assistant response.
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_texts = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_texts[0].strip()


# ---------------------------------------------------------------------------
# Argument parsing / main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Visual understanding (image chat) demo for OmniGen2 / SGT-Gen2. "
            "Reproduces scripts/infra/example_understanding.sh by default."
        )
    )

    # Inputs (defaults match scripts/infra/example_understanding.sh).
    parser.add_argument(
        "--instruction",
        type=str,
        default="Describe this image.",
        help="Natural-language question about the input image.",
    )
    parser.add_argument(
        "--input_image_path",
        type=str,
        nargs="+",
        default=["../assets/pipe.png"],
        help="Path(s) to input image(s).",
    )

    # Model paths.
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
            "expects `<checkpoint_dir>/transformer/text_encoder` to exist, "
            "mirroring the layout used by scripts/infer_edit.py. Pass an "
            "empty string to skip and use the base model."
        ),
    )

    # Generation parameters.
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Enable sampling (default is greedy decoding).",
    )

    # Runtime.
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

    # Optional: dump answer to a file.
    parser.add_argument(
        "--output_text_path",
        type=str,
        default=None,
        help="If set, also save the generated answer text to this file.",
    )
    return parser.parse_args()


def load_images(paths):
    """Load one or more images into PIL.Image (RGB) objects."""
    if isinstance(paths, str):
        paths = [paths]

    # Support passing a single directory -> load every file inside.
    if len(paths) == 1 and os.path.isdir(paths[0]):
        paths = [
            os.path.join(paths[0], f)
            for f in sorted(os.listdir(paths[0]))
        ]

    images = []
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Input image not found: {p}")
        img = Image.open(p).convert("RGB")
        img = ImageOps.exif_transpose(img)
        images.append(img)
    return images


def main():
    args = parse_args()

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    weight_dtype = dtype_map[args.dtype]

    print("=" * 60)
    print("OmniGen2 visual understanding demo")
    print(f"  instruction    : {args.instruction}")
    print(f"  input_image(s) : {args.input_image_path}")
    print(f"  model_path     : {args.model_path}")
    print(f"  checkpoint_dir : {args.checkpoint_dir}")
    print(f"  device/dtype   : {device} / {args.dtype}")
    print("=" * 60)

    images = load_images(args.input_image_path)
    mllm, processor = load_mllm_and_processor(args, device, weight_dtype)

    answer = understand_image(
        mllm,
        processor,
        images,
        args.instruction,
        device,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
    )

    print("\n----- Answer -----")
    print(answer)
    print("------------------\n")

    if args.output_text_path:
        os.makedirs(
            os.path.dirname(os.path.abspath(args.output_text_path)) or ".",
            exist_ok=True,
        )
        with open(args.output_text_path, "w", encoding="utf-8") as f:
            f.write(answer)
        print(f"Saved answer to: {args.output_text_path}")


if __name__ == "__main__":
    main()
