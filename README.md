# SGT: Semantic Generative Tuning for Unified Multimodal Models

<div align="center">

[![Project Page](https://img.shields.io/badge/🌐_Project_Page-Visit-6366f1?style=for-the-badge)](https://song2yu.github.io/sgt-project-page/)
[![Paper](https://img.shields.io/badge/📄_Paper-PDF-8b5cf6?style=for-the-badge)](#)
[![ECCV 2026](https://img.shields.io/badge/🏆_ECCV-2026-ec4899?style=for-the-badge)](#)

</div>

---

## Overview

**SGT (Semantic Generative Tuning)** is the first systematic investigation into generative post-training for Unified Multimodal Models (UMMs). By leveraging **image segmentation as a generative proxy**, SGT bridges the gap between visual understanding and generation, enabling true synergy between the two capabilities within a single architecture.

### Key Results

| Model | CV-Bench↑ | GenEval↑ | GEdit↑ |
|-------|-----------|----------|--------|
| BAGEL (baseline) | 73.21 | 78.21 | 6.52 |
| **SGT-BAGEL (ours)** | **79.23 (+6.02%)** | **80.95** | **6.94** |
| OmniGen2 (baseline) | 65.94 | 76.58 | 6.63 |
| **SGT-Gen2 (ours)** | **66.91** | **78.86** | **6.83** |

---

## Why SGT?

Existing UMMs optimize understanding and generation independently — this leads to misaligned representations and missed synergies. Previous pixel-level alignment methods (ReCA, ROSS, GenHancer) over-emphasize texture and fail to provide structural semantic guidance.

SGT takes a different approach: use **high-level semantic segmentation** as the generative training objective. This simple yet effective proxy:

- ✅ Improves multimodal comprehension (vision-centric reasoning, spatial understanding, hallucination resistance)
- ✅ Enhances generative spatial fidelity
- ✅ Is architecture-agnostic — validated on both BAGEL (7B+7B) and OmniGen2 (3B+4B)
- ✅ Scales monotonically with more segmentation data

---

## Three Key Observations

1. **High-level semantic tasks dominate** — Segmentation consistently outperforms depth estimation and edge detection as a proxy task.
2. **Visual supervision enhances perception, not reasoning** — SGT improves vision-centric and spatial tasks; math/chart reasoning remains unaffected.
3. **Spatial fidelity improves universally** — All proxy levels improve positional generation; segmentation leads overall.

---

## Project Page

👉 **[https://song2yu.github.io/sgt-project-page/](https://song2yu.github.io/sgt-project-page/)**

The project page features:
- Full-screen animated gallery of SGT-generated images
- Interactive results tables with ablation studies
- Method overview and mechanistic analysis

---

## Training Data

SGT uses **190k segmentation samples from SAM** alongside standard VQA SFT data.  
Optimal batch ratio: **2:1 (Segmentation : VQA)**.

| Data Source | Samples |
|-------------|---------|
| SGT — Segmentation (SAM) | 190k |
| General VQA | 180k |
| Doc / Chart / Screen | 103k |
| Math / Reasoning | 101k |
| Language | 72k |
| General OCR | 45k |
| **Total** | **~691k** |

---

## Citation

```bibtex
@inproceedings{anonymous2026sgt,
  title     = {Semantic Generative Tuning for Unified Multimodal Models},
  author    = {Anonymous Authors},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026},
  note      = {Submission #3064}
}
```

---

<div align="center">
  <a href="https://song2yu.github.io/sgt-project-page/">
    <img src="https://img.shields.io/badge/🌐_Visit_Project_Page-song2yu.github.io/sgt--project--page-6366f1?style=flat-square&labelColor=1e1b4b" alt="Project Page"/>
  </a>
</div>
