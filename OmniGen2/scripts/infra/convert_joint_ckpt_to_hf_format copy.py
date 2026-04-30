import dotenv
dotenv.load_dotenv(override=True)

import argparse
import os
from omegaconf import OmegaConf

import torch
# Change 1: drop init_empty_weights; it is no longer needed
# from accelerate import init_empty_weights 

from peft import LoraConfig
from peft.utils import get_peft_model_state_dict

from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel
from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
from transformers import Qwen2_5_VLModel as TextEncoder

# This class is never instantiated by the script, but it documents how the FSDP checkpoint is laid out
class OmniGenJointModel(torch.nn.Module):
    def __init__(self, transformer, text_encoder):
        super().__init__()
        self.model = transformer
        self.text_encoder = text_encoder

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
    
    # Change 2: initialize directly, without using init_empty_weights
    # This allocates every parameter on CPU (random init) and avoids the meta-tensor issue
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
    # Change 3: load from pretrained directly, without using init_empty_weights
    # This downloads/loads the original Qwen weights, which is the correct choice because your checkpoint may omit frozen-layer parameters
    text_encoder = TextEncoder.from_pretrained(
        conf.model.pretrained_text_encoder_model_name_or_path
    )

    # 3. Load checkpoint
    print(f"Loading checkpoint from {model_path}...")
    # state_dict = torch.load(model_path, map_location="cpu") 
    if model_path.endswith('.safetensors'):
        from safetensors.torch import load_file
        state_dict = load_file(model_path)
    else:
        state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    if "module" in state_dict:
        state_dict = state_dict["module"]

    # 4. Split the parameter dictionary
    transformer_state_dict = {}
    text_encoder_state_dict = {}

    print("Splitting state dict into Transformer and Text Encoder...")
    for k, v in state_dict.items():
        if k.startswith("transformer."):
            transformer_state_dict[k[len("transformer."):]] = v
        elif k.startswith("module.transformer."): 
            transformer_state_dict[k[len("module.transformer."):]] = v
        
        elif k.startswith("text_encoder."):
            text_encoder_state_dict[k[len("text_encoder."):]] = v
        elif k.startswith("module.text_encoder."):
            text_encoder_state_dict[k[len("module.text_encoder."):]] = v
        else:
            pass

    print(f"Transformer keys: {len(transformer_state_dict)}")
    print(f"Text Encoder keys: {len(text_encoder_state_dict)}")

    # 5. Load and save the transformer
    print("Loading Transformer weights...")
    # Change 4: drop `assign=True` and use a standard `load_state_dict`
    m, u = transformer.load_state_dict(transformer_state_dict, strict=False)
    print(f"[Transformer] Missing keys: {len(m)}, Unexpected keys: {len(u)}")
    
    # If there are missing keys, print the first few to check whether it is normal
    if len(m) > 0:
        print(f"Sample missing keys: {m[:5]}")

    transformer_save_path = os.path.join(save_root, "transformer")
    print(f"Saving Transformer to {transformer_save_path}...")
    
    if conf.train.get('lora_ft', False):
        transformer_lora_layers = get_peft_model_state_dict(transformer)
        OmniGen2Pipeline.save_lora_weights(
            save_directory=transformer_save_path,
            transformer_lora_layers=transformer_lora_layers,
        )
    else:
        transformer.save_pretrained(transformer_save_path)

    # 6. Load and save the text encoder
    if len(text_encoder_state_dict) > 0:
        print("Loading Text Encoder weights...")
        # Change 5: drop `assign=True`
        m, u = text_encoder.load_state_dict(text_encoder_state_dict, strict=False)
        print(f"[Text Encoder] Missing keys: {len(m)}, Unexpected keys: {len(u)}")
        
        # Missing keys here may be normal (e.g. partial training), because the base weights are already loaded during init
        
        text_encoder_save_path = os.path.join(save_root, "text_encoder")
        print(f"Saving Text Encoder to {text_encoder_save_path}...")
        text_encoder.save_pretrained(text_encoder_save_path)
    else:
        print("Warning: No Text Encoder weights found in checkpoint! Skipping save.")

    print("Conversion finished successfully.")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)