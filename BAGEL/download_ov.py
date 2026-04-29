#!/usr/bin/env python3
"""
Partial download script for the LLaVA-OneVision Single-Image dataset.
Downloads 500k samples following the original category ratios.

Two run modes are supported:
  1. Full download (default):   python download_ov.py
  2. Fix missing data:          python download_ov.py --fix-missing
"""

import os
import sys

# ============================================================
# IMPORTANT: Disable hf_xet BEFORE importing huggingface_hub / datasets
# (hf_xet panics on restricted filesystems when it can't write logs.)
# ============================================================
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_XET_DISABLE"] = "1"
os.environ.setdefault("HF_XET_CACHE", "/tmp/hf_xet_cache")
os.makedirs(os.environ["HF_XET_CACHE"], exist_ok=True)

# Optional: use HF-Mirror (uncomment if needed)
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import io
import json
import random
import argparse
from typing import Dict, List

from tqdm import tqdm
from datasets import load_dataset
from PIL import Image

# Set random seed for reproducibility
random.seed(42)

# ============================================================
# Configuration
# ============================================================
OUTPUT_DIR = "./data/LLaVA-OneVision-SGT/llava_onevision_balanced_500k/"
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
ANNOTATION_DIR = os.path.join(OUTPUT_DIR, "annotations")

TOTAL_SAMPLES = 500000

CATEGORY_RATIOS = {
    "General": 0.361,
    "Doc_Chart_Screen": 0.206,
    "Math_Reasoning": 0.201,
    "General_OCR": 0.089,
    "Language": 0.143,
}

HF_DATASETS = {
    "General": {
        "vision_flan(filtered)": 5000,
        "cambrian(filtered)": 5100,
    },
    "Doc_Chart_Screen": {
        "chartqa(cauldron,llava_format)": 4000,
        "ai2d(gpt4v)": 4000,
        "ai2d(internvl)": 4000,
        "dvqa(cauldron,llava_format)": 4000,
        "docvqa(cauldron,llava_format)": 2600,
        "infographic_vqa_llava_format": 2000,
    },
    "Math_Reasoning": {
        "geo170k(qa)": 5000,
        "geo170k(align)": 5000,
        "GeoQA+(MathV360K)": 3000,
        "TabMWP(MathV360K)": 3000,
        "mathqa": 2100,
        "UniGeo(MathV360K)": 2000,
    },
    "General_OCR": {
        "textcaps": 3000,
        "hme100k": 2000,
        "k12_printing": 1500,
        "textocr(gpt4v)": 1400,
        "rendered_text(cauldron)": 1000,
    },
    "Language": {
        "magpie_pro(l3_80b_st)": 7000,
        "magpie_pro(qwen2_72b_st)": 7300,
    },
}

HF_REPO = "lmms-lab/LLaVA-OneVision-Data"

FIX_MISSING_CATEGORY = "General"
FIX_MISSING_DATASETS = {
    "sharegpt4o": 2000,
    "sharegpt4v(sam)": 1500,
    "allava_instruct_laion4v": 1600,
}


# ============================================================
# Utility functions
# ============================================================
def create_directories():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(ANNOTATION_DIR, exist_ok=True)
    for category in HF_DATASETS.keys():
        os.makedirs(os.path.join(IMAGE_DIR, category), exist_ok=True)


def save_image(image, save_path):
    try:
        if isinstance(image, Image.Image):
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            image.save(save_path, "JPEG", quality=95)
            return True
        elif isinstance(image, bytes):
            img = Image.open(io.BytesIO(image))
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            img.save(save_path, "JPEG", quality=95)
            return True
        elif isinstance(image, dict) and 'bytes' in image:
            img = Image.open(io.BytesIO(image['bytes']))
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            img.save(save_path, "JPEG", quality=95)
            return True
        else:
            print(f"Unknown image format: {type(image)}")
            return False
    except Exception as e:
        print(f"Failed to save image: {e}")
        return False


def sample_dataset(dataset, target_size: int) -> List:
    total_size = len(dataset)
    if total_size <= target_size:
        indices = list(range(total_size))
    else:
        indices = random.sample(range(total_size), target_size)
    return indices


def download_and_process_subset(
    subset_name: str,
    target_samples: int,
    category: str,
) -> List[Dict]:
    print(f"\n  Downloading {subset_name} (target: {target_samples})...")

    try:
        dataset = load_dataset(
            HF_REPO,
            subset_name,
            split="train",
        )

        total_size = len(dataset)
        print(f"    Dataset size: {total_size}")

        indices = sample_dataset(dataset, target_samples)
        print(f"    Sample count: {len(indices)}")

        safe_subset_name = subset_name.replace('(', '_').replace(')', '_').replace(',', '_')
        image_dir = os.path.join(IMAGE_DIR, category, safe_subset_name)
        os.makedirs(image_dir, exist_ok=True)

        processed_data = []
        success_count = 0

        for idx in tqdm(indices, desc=f"    Processing {subset_name[:30]}"):
            try:
                item = dataset[idx]

                item_id = f"{safe_subset_name}_{idx}"
                image_filename = f"{item_id}.jpg"
                image_path = os.path.join(image_dir, image_filename)
                relative_image_path = os.path.join(category, safe_subset_name, image_filename)

                if "image" in item and item["image"] is not None:
                    if not os.path.exists(image_path):
                        if not save_image(item["image"], image_path):
                            continue
                    success_count += 1
                else:
                    relative_image_path = None

                processed_item = {
                    "id": item_id,
                    "category": category,
                    "dataset": subset_name,
                    "image": relative_image_path,
                }

                if "conversations" in item:
                    processed_item["conversations"] = item["conversations"]
                elif "messages" in item:
                    processed_item["conversations"] = item["messages"]

                for key in ["question", "answer", "text", "caption"]:
                    if key in item and item[key]:
                        processed_item[key] = item[key]

                processed_data.append(processed_item)

            except Exception as e:
                print(f"    Failed to process item {idx}: {e}")
                continue

        print(f"    Done! Processed: {len(processed_data)}, images saved: {success_count}")
        return processed_data

    except Exception as e:
        print(f"    Failed to load dataset: {e}")
        return []


def verify_config():
    print("=" * 60)
    print("Configuration check:")
    print("=" * 60)

    total = 0
    for category, subsets in HF_DATASETS.items():
        category_total = sum(subsets.values())
        expected = int(TOTAL_SAMPLES * CATEGORY_RATIOS[category])
        total += category_total
        print(f"  {category}: {category_total} (target: {expected}, ratio: {CATEGORY_RATIOS[category]*100:.1f}%)")

    print(f"\n  Total: {total} (target: {TOTAL_SAMPLES})")
    print("=" * 60)

    return total == TOTAL_SAMPLES


# ============================================================
# Full download pipeline
# ============================================================
def run_full_download():
    print("=" * 60)
    print("LLaVA-OneVision Single-Image dataset download tool")
    print(f"Target: {TOTAL_SAMPLES} samples, following original ratios")
    print(f"Output dir: {OUTPUT_DIR}")
    print("=" * 60)

    if not verify_config():
        print("Warning: configured totals do not match target, continuing anyway...")

    create_directories()

    total_stats = {}
    all_data = {}

    for category, subsets in HF_DATASETS.items():
        print(f"\n{'='*60}")
        print(f"Processing category: {category} (target: {int(TOTAL_SAMPLES * CATEGORY_RATIOS[category])})")
        print(f"{'='*60}")

        category_data = []

        for subset_name, target_samples in subsets.items():
            data = download_and_process_subset(
                subset_name=subset_name,
                target_samples=target_samples,
                category=category,
            )
            category_data.extend(data)

        all_data[category] = category_data
        total_stats[category] = len(category_data)

        category_file = os.path.join(ANNOTATION_DIR, f"{category}.json")
        with open(category_file, 'w', encoding='utf-8') as f:
            json.dump(category_data, f, ensure_ascii=False, indent=2)

        print(f"\nCategory {category} done, {len(category_data)} samples total")

    all_merged = []
    for category_data in all_data.values():
        all_merged.extend(category_data)

    merged_file = os.path.join(ANNOTATION_DIR, "all_data.json")
    with open(merged_file, 'w', encoding='utf-8') as f:
        json.dump(all_merged, f, ensure_ascii=False, indent=2)

    summary = {
        "total_samples": len(all_merged),
        "target_samples": TOTAL_SAMPLES,
        "category_stats": {},
    }

    print("\n" + "=" * 60)
    print("Download complete! Statistics:")
    print("=" * 60)

    total = 0
    for category, count in total_stats.items():
        expected = int(TOTAL_SAMPLES * CATEGORY_RATIOS[category])
        actual_ratio = count / len(all_merged) * 100 if all_merged else 0
        print(f"  {category}: {count:,} samples ({actual_ratio:.1f}%, target: {CATEGORY_RATIOS[category]*100:.1f}%)")
        total += count
        summary["category_stats"][category] = {
            "count": count,
            "expected": expected,
            "ratio": actual_ratio,
        }

    print(f"\n  Total: {total:,} samples")
    print(f"  Output dir: {os.path.abspath(OUTPUT_DIR)}")

    summary_file = os.path.join(OUTPUT_DIR, "summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"  Summary file: {summary_file}")


# ============================================================
# Fix-missing pipeline
# ============================================================
def run_fix_missing():
    print("=" * 60)
    print("Fix-missing: downloading replacement General-category data")
    print(f"Replacement datasets: {list(FIX_MISSING_DATASETS.keys())}")
    print(f"Target total: {sum(FIX_MISSING_DATASETS.values())}")
    print("=" * 60)

    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(ANNOTATION_DIR, exist_ok=True)
    os.makedirs(os.path.join(IMAGE_DIR, FIX_MISSING_CATEGORY), exist_ok=True)

    all_new_data = []
    for subset_name, target_samples in FIX_MISSING_DATASETS.items():
        data = download_and_process_subset(
            subset_name=subset_name,
            target_samples=target_samples,
            category=FIX_MISSING_CATEGORY,
        )
        all_new_data.extend(data)

    print(f"\nNewly downloaded: {len(all_new_data)} samples")

    general_file = os.path.join(ANNOTATION_DIR, f"{FIX_MISSING_CATEGORY}.json")
    if os.path.exists(general_file):
        with open(general_file, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
        print(f"Existing {FIX_MISSING_CATEGORY} data: {len(existing_data)} samples")
    else:
        existing_data = []
        print(f"{FIX_MISSING_CATEGORY}.json not found, creating a new file")

    combined_data = existing_data + all_new_data
    with open(general_file, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=2)
    print(f"Updated {FIX_MISSING_CATEGORY} data: {len(combined_data)} samples")

    all_data_file = os.path.join(ANNOTATION_DIR, "all_data.json")
    if os.path.exists(all_data_file):
        with open(all_data_file, 'r', encoding='utf-8') as f:
            all_data = json.load(f)
        print(f"Existing all_data: {len(all_data)} samples")

        all_data.extend(all_new_data)
        with open(all_data_file, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)
        print(f"Updated all_data: {len(all_data)} samples")
    else:
        print("all_data.json not found, skipping update")

    print("\n" + "=" * 60)
    print("Fix-missing done!")
    print("=" * 60)


# ============================================================
# Entry point
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="LLaVA-OneVision dataset download / fix-missing tool",
    )
    parser.add_argument(
        "--fix-missing",
        action="store_true",
        help="Run the fix-missing pipeline.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.fix_missing:
        run_fix_missing()
    else:
        run_full_download()


if __name__ == "__main__":
    main()