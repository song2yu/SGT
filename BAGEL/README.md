---
license: apache-2.0
pipeline_tag: any-to-any
library_name: bagel-mot
tags:
  - sgt
  - semantic-generative-tuning
  - unified-multimodal
  - image-segmentation
  - visual-understanding
  - visual-generation
---

# SGT: Semantic Generative Tuning for Unified Multimodal Models

This repository hosts checkpoints fine-tuned with **Semantic Generative Tuning (SGT)** — a training
paradigm that couples visual *understanding* and *generation* in Unified Multimodal Models (UMMs)
by using **image segmentation as a generative proxy**.

> Unified multimodal models typically optimize understanding and generation with *misaligned*
> objectives (sparse text tokens vs. dense pixel targets), which isolates the two capabilities.
> SGT introduces segmentation — a **high-level semantic task** — as a unified generative objective
> that aligns the two branches, improves feature linear separability, and optimizes visual-textual
> attention allocation.

## 🧠 Method Overview

SGT reformulates classical visual tasks as generative proxies and establishes a **hierarchical
taxonomy** (low-/mid-/high-level). Extensive experiments show that **high-level semantic tasks
(e.g. image segmentation) are the optimal proxy**, outperforming depth, edge, reconstruction and
MAE/inpainting for synergizing understanding and generation.

Key findings:

1. **High-level > low-level**: segmentation gives larger gains in visual understanding
   than depth / edge / pixel reconstruction.
2. **Perception, not reasoning**: visual supervision mainly strengthens perception
   (spatial, hallucination, vision-centric, general VQA), rather than abstract reasoning (e.g. math, chart)
3. **Architecture-agnostic**: the gains hold for both **BAGEL** and **OmniGen2**.

## 📦 Released Artifacts

| Repo | Type | Base Model | Content |
|---|---|---|---|
| [`Two-hot/SGT-BAGEL`](https://huggingface.co/Two-hot/SGT-BAGEL)   | model   | BAGEL-7B-MoT   | SGT fine-tuned BAGEL checkpoint |
| [`Two-hot/SGT-Gen2`](https://huggingface.co/Two-hot/SGT-Gen2)     | model   | OmniGen2       | SGT fine-tuned OmniGen2 checkpoint (transformer/ only) |
| [`Two-hot/SAM-SGT`](https://huggingface.co/datasets/Two-hot/SAM-SGT) | dataset | —            | Segmentation training data (tar-sharded) used by SGT |

### Use the SAM-SGT dataset

See [`Two-hot/SAM-SGT`](https://huggingface.co/datasets/Two-hot/SAM-SGT) for the data
layout and the extraction instructions.

## 📊 Highlights

- **+6.02%** average gain over BAGEL on the **CV-Bench** evaluation.
- Consistent improvements in **spatial reasoning**, **hallucination resistance**, **vision-centric**, **general VQA**, and **OCR**.
- Generation: gains across **GenEval** dimensions (Position / Color etc.).
- Verified on two representative UMM architectures (**BAGEL**, **OmniGen2**).

## 📝 License

Apache-2.0. Base models remain under their original licenses:
BAGEL (Apache-2.0, based on Qwen2.5-7B + SigLIP + FLUX VAE) and
OmniGen2 (based on Qwen2.5-VL + diffusion transformer).

## ✍️ Citation

If you find this work useful, please cite our paper:

```bibtex
@article{sgt2026,
  title   = {Semantic Generative Tuning for Unified Multimodal Models},
  author  = {Songsong Yu, Yuxin Chen, Ying Shan, and Yanwei Li},
  journal = {arxiv},
  year    = {2026}
}
```