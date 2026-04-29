# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import os
os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import sys

import torch
import json
import re
import random
import numpy as np
from datetime import datetime
from tqdm import tqdm
from datasets import load_dataset
from PIL import Image

# BAGEL-related imports.
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights
from data.transforms import ImageTransform
from data.data_utils import add_special_tokens
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.autoencoder import load_ae
from inferencer import InterleaveInferencer

# Scoring model imports.
from transformers import AutoModelForCausalLM, AutoTokenizer


# ==================== Configuration ====================
class Config:
    # BAGEL model path.
    # checkpoint=os.path.join(model_path, 'ema.safetensors'), # official
    # checkpoint=os.path.join('ckpt/BAGEL-RecA/model_bf16.safetensors'), # ckpt/BAGEL-RecA/model_bf16.safetensors
    BAGEL_MODEL_PATH = "ckpt/BAGEL-7B-MoT"
    # BAGEL_CHECKPOINT = os.path.join(BAGEL_MODEL_PATH, 'ema.safetensors')
    BAGEL_CHECKPOINT = "ckpt/BAGEL-SGT/ema.safetensors"
    
    # Qwen LLM model (used for scoring).
    SCORING_MODEL_PATH = "Qwen/Qwen2-0.5B-Instruct"
    
    # GPU configuration.
    BAGEL_GPU_ID = 0  # GPU index used by BAGEL.
    MAX_MEM_PER_GPU = "40GiB"
    
    # Output configuration.
    OUTPUT_FILE = "bagel_panoptic-100k_3k_results.json" # bagel_panoptic_3k_results    bagel_official_results
    MAX_SAMPLES = None  # None = evaluate everything; otherwise a limit (e.g. 100).
    
    # Random seed.
    SEED = 42


# ==================== Seed handling ====================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==================== Load the BAGEL model ====================
def load_bagel_model(model_path, checkpoint_path, gpu_id=0, max_mem="80GiB"):
    """Load the BAGEL model."""
    print(f"Loading BAGEL model: {model_path}")
    print(f"Checkpoint: {checkpoint_path}")
    
    # LLM config
    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"
    
    # ViT config
    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1
    
    # VAE loading
    vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))
    
    # Bagel config
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
    
    # Initialise the model.
    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)
    
    # Tokenizer
    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
    
    # Image Transform
    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, 224, 14)
    
    # Device map
    device_map = infer_auto_device_map(
        model,
        max_memory={gpu_id: max_mem},
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    )
    print(f"Device map: {device_map}")
    
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
        first_device = device_map.get(same_device_modules[0], f"cuda:{gpu_id}")
        for k in same_device_modules:
            if k in device_map:
                device_map[k] = first_device
            else:
                device_map[k] = f"cuda:{gpu_id}"
    else:
        first_device = device_map.get(same_device_modules[0])
        for k in same_device_modules:
            if k in device_map:
                device_map[k] = first_device
    
    # Load checkpoint.
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=checkpoint_path,
        device_map="auto",
        offload_buffers=False,
        dtype=torch.bfloat16,
        force_hooks=True,
    )
    
    model = model.eval()
    
    # Build the inferencer.
    inferencer = InterleaveInferencer(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids
    )
    
    print("BAGEL model loaded!")
    return inferencer


# ==================== Load the scoring LLM ====================
def load_scoring_llm(model_path):
    """Load the Qwen LLM used for scoring."""
    print(f"Loading scoring model: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    print("Scoring model loaded!")
    return model, tokenizer


# ==================== BAGEL inference ====================
def bagel_inference(inferencer, image, question):
    """Run image + text inference with BAGEL."""
    
    inference_hyper = dict(
        max_think_token_n=1000,
        do_sample=False,
    )
    
    output_dict = inferencer(
        image=image,
        text=question,
        understanding_output=True,
        **inference_hyper
    )
    
    return output_dict['text']


# ==================== LLM scorer ====================
class LLMScorer:
    """Use a local Qwen LLM to score generated answers."""
    
    SCORING_PROMPT = """You are a professional answer evaluator. Compare the
reference answer with the model-generated answer and assign a score from 0 to 10.

Scoring rubric:
- 10 : fully correct; matches the reference answer.
- 7-9: mostly correct; the key information is accurate.
- 4-6: partially correct; some accurate info, some mistakes.
- 1-3: mostly wrong; only a few accurate details.
- 0  : completely wrong or unrelated.

Question: {question}

Reference answer: {reference}

Model answer: {prediction}

Return ONLY a single integer between 0 and 10. Do not return anything else.
Score: """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    def score(self, question, reference, prediction):
        """Score the model output."""
        prompt = self.SCORING_PROMPT.format(
            question=question,
            reference=reference,
            prediction=prediction
        )
        
        messages = [
            {"role": "system", "content": "You are a professional scoring assistant. Return only a numeric score."},
            {"role": "user", "content": prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False
            )
        
        # Decode the output.
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        # Extract the numeric score.
        try:
            numbers = re.findall(r'\d+', response)
            if numbers:
                score = int(numbers[0])
                return max(0, min(10, score))
            else:
                return 5
        except:
            return 5


# ==================== Main evaluation function ====================
def evaluate_dataset():
    """Run the full evaluation."""
    
    # Set the random seed.
    set_seed(Config.SEED)
    
    # 1. Load the dataset.
    print("=" * 50)
    print("Loading dataset ...")
    ds = load_dataset("xiaoyuanliu/conflict_vis", split="train")
    print(f"Dataset size: {len(ds)}")
    print("=" * 50)
    
    # 2. Load the BAGEL model (for image + text inference).
    inferencer = load_bagel_model(
        model_path=Config.BAGEL_MODEL_PATH,
        checkpoint_path=Config.BAGEL_CHECKPOINT,
        gpu_id=Config.BAGEL_GPU_ID,
        max_mem=Config.MAX_MEM_PER_GPU
    )
    
    # 3. Load the Qwen LLM used for scoring.
    scoring_model, scoring_tokenizer = load_scoring_llm(Config.SCORING_MODEL_PATH)
    
    # 4. Initialise the scorer.
    scorer = LLMScorer(scoring_model, scoring_tokenizer)
    
    # 5. Decide how many samples to evaluate.
    num_samples = len(ds) if Config.MAX_SAMPLES is None else min(Config.MAX_SAMPLES, len(ds))
    print(f"Evaluating {num_samples} samples.")
    print("=" * 50)
    
    # 6. Evaluation loop.
    results = []
    total_score = 0
    success_count = 0
    
    for idx in tqdm(range(num_samples), desc="evaluating"):
        sample = ds[idx]
        
        try:
            # Pull the sample's fields.
            image = sample['images']  # already a PIL image
            question = sample['question']
            reference = sample['answer']
            sample_id = sample.get('id', str(idx))
            
            # Make sure the image is in RGB mode.
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # BAGEL inference.
            prediction = bagel_inference(inferencer, image, question)
            
            # LLM scoring.
            score = scorer.score(question, reference, prediction)
            
            # Record the result.
            result = {
                "id": sample_id,
                "question": question,
                "reference": reference,
                "prediction": prediction,
                "score": score,
                "question_type": sample.get('question_type', ''),
                "conflict_target": sample.get('conflict_target', '')
            }
            results.append(result)
            
            total_score += score
            success_count += 1
            
        except Exception as e:
            print(f"\nSample {idx} failed: {e}")
            results.append({
                "id": sample.get('id', str(idx)),
                "error": str(e),
                "score": 0
            })
    
    # 7. Compute summary statistics.
    avg_score = total_score / success_count if success_count > 0 else 0
    
    # Aggregate by question type.
    type_scores = {}
    for r in results:
        if 'error' not in r:
            q_type = r.get('question_type', 'unknown')
            if q_type not in type_scores:
                type_scores[q_type] = {'total': 0, 'count': 0}
            type_scores[q_type]['total'] += r['score']
            type_scores[q_type]['count'] += 1
    
    for q_type in type_scores:
        type_scores[q_type]['avg'] = (
            type_scores[q_type]['total'] / type_scores[q_type]['count']
            if type_scores[q_type]['count'] > 0 else 0
        )
    
    # Aggregate by conflict target.
    conflict_scores = {}
    for r in results:
        if 'error' not in r:
            c_target = r.get('conflict_target', 'unknown')
            if c_target not in conflict_scores:
                conflict_scores[c_target] = {'total': 0, 'count': 0}
            conflict_scores[c_target]['total'] += r['score']
            conflict_scores[c_target]['count'] += 1
    
    for c_target in conflict_scores:
        conflict_scores[c_target]['avg'] = (
            conflict_scores[c_target]['total'] / conflict_scores[c_target]['count']
            if conflict_scores[c_target]['count'] > 0 else 0
        )
    
    # 8. Collect summary results.
    summary = {
        "evaluation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": "BAGEL",
        "model_path": Config.BAGEL_MODEL_PATH,
        "checkpoint": Config.BAGEL_CHECKPOINT,
        "scoring_model": Config.SCORING_MODEL_PATH,
        "total_samples": num_samples,
        "success_count": success_count,
        "average_score": round(avg_score, 4),
        "score_by_question_type": type_scores,
        "score_by_conflict_target": conflict_scores
    }
    
    # 9. Persist results.
    output_data = {
        "summary": summary,
        "results": results
    }
    
    with open(Config.OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    # 10. Log results.
    print("\n" + "=" * 50)
    print("Evaluation complete!")
    print("=" * 50)
    print(f"Model: BAGEL")
    print(f"Total samples: {num_samples}")
    print(f"Successful: {success_count}")
    print(f"Average score: {avg_score:.4f} / 10")
    print("-" * 50)
    print("Breakdown by question type:")
    for q_type, stats in type_scores.items():
        print(f"  {q_type}: {stats['avg']:.4f} (n={stats['count']})")
    print("-" * 50)
    print("Breakdown by conflict target:")
    for c_target, stats in conflict_scores.items():
        print(f"  {c_target}: {stats['avg']:.4f} (n={stats['count']})")
    print("-" * 50)
    print(f"Results saved to: {Config.OUTPUT_FILE}")
    
    return summary, results


# ==================== Entry point ====================
if __name__ == "__main__":
    evaluate_dataset()