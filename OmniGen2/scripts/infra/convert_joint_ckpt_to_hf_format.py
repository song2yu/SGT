import dotenv
dotenv.load_dotenv(override=True)

import argparse
import os
from omegaconf import OmegaConf

import torch

from peft import LoraConfig
from peft.utils import get_peft_model_state_dict

from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel
from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
from transformers import Qwen2_5_VLModel as TextEncoder


class OmniGenJointModel(torch.nn.Module):
    """Used only to help understand the FSDP checkpoint layout"""
    def __init__(self, transformer, text_encoder):
        super().__init__()
        self.model = transformer          # The transformer corresponds to self.model
        self.text_encoder = text_encoder  # The text encoder corresponds to self.text_encoder
        self.lm_head = text_encoder.lm_head
        
    def forward(self, *args, **kwargs):
        pass


def main(args):
    config_path = args.config_path
    model_path = args.model_path
    save_root = args.save_path 

    conf = OmegaConf.load(config_path)
    arch_opt = conf.model.arch_opt

    # 1. Initialize the transformer structure
    arch_opt = OmegaConf.to_object(arch_opt)
    for key, value in arch_opt.items():
        if isinstance(value, list):
            arch_opt[key] = tuple(value)

    print("Initializing Transformer structure (on CPU)...")
    transformer = OmniGen2Transformer2DModel(**arch_opt)

    if conf.train.get('lora_ft', False):
        print("Adding LoRA adapter...")
        target_modules = ["to_k", "to_q", "to_v", "to_out.0"]
        lora_config = LoraConfig(
            r=conf.train.lora_rank,
            lora_alpha=conf.train.lora_rank,
            lora_dropout=conf.train.lora_dropout,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        transformer.add_adapter(lora_config)

    # 2. Initialize the text-encoder structure
    print(f"Initializing Text Encoder structure ({conf.model.pretrained_text_encoder_model_name_or_path})...")
    text_encoder = TextEncoder.from_pretrained(
        conf.model.pretrained_text_encoder_model_name_or_path
    )

    # 3. Load checkpoint
    print(f"Loading checkpoint from {model_path}...")
    if model_path.endswith('.safetensors'):
        from safetensors.torch import load_file
        state_dict = load_file(model_path)
    else:
        state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    
    # Handle a possibly nested structure
    if "module" in state_dict:
        state_dict = state_dict["module"]

    # [Debug] print the first 15 keys to inspect the actual prefix format
    print("\n" + "="*50)
    print("Sample keys from checkpoint:")
    print("="*50)
    for i, k in enumerate(state_dict.keys()):
        print(f"  {k}")
        if i >= 15:
            print(f"  ... (total {len(state_dict)} keys)")
            break
    print("="*50 + "\n")

    # 4. Split the parameter dictionary
    transformer_state_dict = {}
    text_encoder_state_dict = {}
    unmatched_keys = []

    print("Splitting state dict into Transformer and Text Encoder...")
    for k, v in state_dict.items():
        # Important: the transformer is stored as `self.model` inside OmniGenJointModel
        if k.startswith("model."):
            transformer_state_dict[k[len("model."):]] = v
        elif k.startswith("module.model."): 
            transformer_state_dict[k[len("module.model."):]] = v
        # Also support the legacy `transformer.` prefix just in case
        elif k.startswith("transformer."):
            transformer_state_dict[k[len("transformer."):]] = v
        elif k.startswith("module.transformer."): 
            transformer_state_dict[k[len("module.transformer."):]] = v
        # `text_encoder.` prefix
        elif k.startswith("text_encoder."):
            text_encoder_state_dict[k[len("text_encoder."):]] = v
        elif k.startswith("module.text_encoder."):
            text_encoder_state_dict[k[len("module.text_encoder."):]] = v
        else:
            unmatched_keys.append(k)

    print(f"Transformer keys: {len(transformer_state_dict)}")
    print(f"Text Encoder keys: {len(text_encoder_state_dict)}")
    
    if len(unmatched_keys) > 0:
        print(f"Unmatched keys: {len(unmatched_keys)}")
        print(f"Sample unmatched keys: {unmatched_keys[:5]}")

    # 5. Load and save the transformer
    if len(transformer_state_dict) > 0:
        print("\nLoading Transformer weights...")
        m, u = transformer.load_state_dict(transformer_state_dict, strict=False)
        print(f"[Transformer] Missing keys: {len(m)}, Unexpected keys: {len(u)}")
        
        if len(m) > 0:
            print(f"Sample missing keys: {m[:5]}")
        if len(u) > 0:
            print(f"Sample unexpected keys: {u[:5]}")

        transformer_save_path = os.path.join(save_root, "transformer")
        os.makedirs(transformer_save_path, exist_ok=True)
        print(f"Saving Transformer to {transformer_save_path}...")
        
        if conf.train.get('lora_ft', False):
            transformer_lora_layers = get_peft_model_state_dict(transformer)
            OmniGen2Pipeline.save_lora_weights(
                save_directory=transformer_save_path,
                transformer_lora_layers=transformer_lora_layers,
            )
        else:
            transformer.save_pretrained(transformer_save_path)
        
        print("[Transformer] Saved successfully!")
    else:
        print("\n" + "!"*50)
        print("ERROR: No Transformer weights found in checkpoint!")
        print("Please check the checkpoint key prefixes above.")
        print("!"*50)

    # 6. Load and save the text encoder
    print("\n" + "-"*50)
    if len(text_encoder_state_dict) > 0:
        print("Loading Text Encoder weights from checkpoint...")
        m, u = text_encoder.load_state_dict(text_encoder_state_dict, strict=False)
        print(f"[Text Encoder] Missing keys: {len(m)}, Unexpected keys: {len(u)}")
        
        if len(m) > 0:
            print(f"Sample missing keys: {m[:5]}")
        if len(u) > 0:
            print(f"Sample unexpected keys: {u[:5]}")
    else:
        print("No Text Encoder weights in checkpoint, using pretrained weights.")

    # Save the text encoder regardless
    text_encoder_save_path = os.path.join(save_root, "text_encoder")
    os.makedirs(text_encoder_save_path, exist_ok=True)
    print(f"Saving Text Encoder to {text_encoder_save_path}...")
    text_encoder.save_pretrained(text_encoder_save_path)
    print("[Text Encoder] Saved successfully!")

    print("\n" + "="*50)
    print("Conversion finished successfully!")
    print("="*50)
    print(f"\nOutput files:")
    print(f"  - Transformer: {transformer_save_path}")
    print(f"  - Text Encoder: {text_encoder_save_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)