# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import torch
import json
import re
from datetime import datetime
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    AutoModelForCausalLM,
    AutoTokenizer
)
from qwen_vl_utils import process_vision_info


# ==================== Configuration ====================
class Config:
    # Qwen-VL model (image + text inference).
    VL_MODEL_PATH = "Qwen/Qwen2-VL-7B-Instruct"
    
    # Qwen LLM model (used for scoring).
    SCORING_MODEL_PATH = "Qwen/Qwen2-7B-Instruct"
    
    # Output configuration.
    OUTPUT_FILE = "evaluation_results.json"
    MAX_SAMPLES = None  # None = evaluate everything; otherwise a limit (e.g. 100).


# ==================== Load the Qwen-VL model ====================
def load_qwen_vl_model(model_path):
    """Load the Qwen2-VL model."""
    print(f"Loading Qwen-VL model: {model_path}")
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        trust_remote_code=True
    )
    
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    print("Qwen-VL model loaded!")
    return model, processor


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


# ==================== Qwen-VL inference ====================
def qwen_vl_inference(model, processor, image, question):
    """Run image + text inference with Qwen-VL."""
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]
    
    # Prepare the inputs.
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    
    # Generate the answer.
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False
        )
    
    # Decode the output.
    generated_ids_trimmed = [
        out_ids[len(in_ids):] 
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    
    return output_text[0]


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
    
    # 1. Load the dataset.
    print("=" * 50)
    print("Loading dataset ...")
    ds = load_dataset("xiaoyuanliu/conflict_vis", split="train")
    print(f"Dataset size: {len(ds)}")
    print("=" * 50)
    
    # 2. Load the Qwen-VL model (for image + text inference).
    vl_model, vl_processor = load_qwen_vl_model(Config.VL_MODEL_PATH)
    
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
        
        # Uncomment below to restrict evaluation to yes/no questions only.
        if sample['question_type'] != 'YN':
            continue
        
        try:
            # Pull the sample's fields.
            image = sample['images']  # already a PIL image
            question = sample['question']
            reference = sample['answer']
            sample_id = sample.get('id', str(idx))
            
            # Qwen-VL inference.
            prediction = qwen_vl_inference(vl_model, vl_processor, image, question)
            
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
        "vl_model": Config.VL_MODEL_PATH,
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