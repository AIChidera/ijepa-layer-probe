# I-JEPA vs DINOv2 for Unsupervised Anomaly Detection: A Layer-Wise Mechanistic Study

**Unsupervised anomaly detection via intermediate representations of joint-embedding predictive architectures.**

> This repository contains the standalone, reproducible Jupyter Notebook required to execute the layer-wise feature extraction, greedy coreset subsampling, and anomaly detection scoring pipelines presented in the manuscript.

---

## Why this work

Pretrained vision models are highly transferable to downstream tasks. But what is less determined is the exact encoder depths at which these models form optimally useful representations. In unsupervised anomaly detection (UAD), the field has blindly assumed that the optimal representations reside in final-layer features.

This work challenges that assumption. I-JEPA's pretraining task does not have the same structure as the discriminative models (like DINOv2) that UAD research has settled on. The pretraining goal of I-JEPA is to bring out globally coherent and predictable representations out of the final block. This pressure compresses the diverse local-patch appearances into a single compact manifold, driving a geometric collapse that contradicts the variance k-NN anomaly scoring requires. 

Intermediate blocks are only indirectly influenced by the pretraining goal. It is only through this distance that they are able to preserve the local geometry necessary to make correct anomaly discriminations. 

The empirical results are conclusive:

| Dataset | I-JEPA Bl-Opt | I-JEPA Bl32 | DINOv2 Bl24 | Gain (Opt vs B32) |
|---|---|---|---|---|
| **MVTec AD (n=15)** | **0.977** (Bl24) | 0.905 | 0.966 | **+7.2 pp** |
| **VisA (n=12)** | **0.956** (Bl24) | 0.752 | 0.934 | **+20.4 pp** |

The perceived inferiority of masked prediction models in prior literature is not a natural property of the architecture. It is an artifact of layer selection. Shift the probe point to an intermediate layer, and the deficit vanishes.

---

## Architecture

```text
Raw Image (448×448)
    │
    ▼ ImageNet Normalization & Patchification
    │
    ├──── Frozen Context Encoder (I-JEPA ViT-H/14)
    │
    ▼ Extract at Target Probe Layer (Block 8, 16, 20, 24, 28, or 32)
    │
    ├──── Spatial Patch-Token Features (1024 tokens)
    │
    ├──► Training: Greedy Farthest-Point Coreset Subsampling ──► Memory Bank M (max 20k vectors)
    │
    └──► Inference: k-NN Search (k=3) against Memory Bank M
                        │
                        ▼ 32×32 Patch Score Map
                        │
        ┌───────────────┴───────────────┐
        ▼                               ▼
  Gaussian Spatial Filter (σ=4.0)   Max Patch Score Pooling
        │                               │
  Bilinear Upsampling                   ▼
        │                           Image-Level Anomaly Score
        ▼ 
  Pixel-Level Anomaly Heatmap
```

**Feature Extraction**: The frozen encoder preserves a full 1024-token spatial grid. We explicitly bypass CLS-token slicing to ensure no spatial patches are deleted.

**Coreset Subsampling**: Random image-count caps aggressively discard normal appearances. Instead, we pool raw features and apply greedy farthest-point subsampling, locking memory by vector count (max 20,000) rather than image count, ensuring all training images are represented.

**Spatial Smoothing**: Bilinear upsampling of a coarse 32×32 grid creates severe blur artifacts. Applying a Gaussian spatial filter (σ = 4.0) mathematically resolves hard edge artifacts *before* upsampling, aggressively tightening pixel-level localization.

---

## Execution Instructions

The entire execution pipeline and environment setup is strictly contained within `Reproducibility Noteboo.ipynb`.

1. **Hardware Constraints:** The evaluation relies on layer-wise k-NN memory banks. A minimum of 16GB VRAM (NVIDIA T4 or equivalent) is required to prevent out-of-memory errors. Multi-GPU execution is natively supported via `nn.DataParallel`.
2. **Environment Initialization:** Run Cell 1 to lock in the required pip dependencies.
3. **Automated Data Retrieval:** Do not manually download or mount datasets. Cell 2 interfaces directly with the anonymized Zenodo vault. It programmatically fetches the VisA dataset, MVTec AD dataset, and the specific I-JEPA ViT-H/14 weights, extracting and routing them into a strict `./workspace/` hierarchy.
4. **Configuration:** To evaluate different permutations, adjust the ablation parameters (`DATASET`, `ENCODER`, `K_NEIGHBORS`) under the Ablation Knobs section in Cell 2.
5. **Execution:** Execute all subsequent cells sequentially. The pipeline bypasses materializing the full embedding matrix to prevent RAM spikes, dynamically generates the Gaussian-smoothed pixel heatmaps, and outputs the raw SVD-based mechanistic metrics.

---

## Datasets

**MVTec AD**: 15 categories (5 texture, 10 object) of high-resolution industrial inspection images.
**VisA**: 12 categories covering complex periodic structures (pasta) and fine-scale structural textures (PCBs). 

Both benchmarks represent a highly diverse spatial frequency profile necessary for mapping encoder depth responses. 

---

## Theoretical grounding

This study bridges the gap between vision transformer representation geometry and anomaly detection, building upon:

**Assran et al. (2023)**: *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture (I-JEPA).* Establishes the non-generative masked prediction objective that inherently drives final-layer geometric collapse.

**Oquab et al. (2024)**: *DINOv2: Learning Robust Visual Features without Supervision.* The discriminatively trained baseline.

**Defard et al. (2021) & Roth et al. (2022)**: *PaDiM & PatchCore.* Establishes the standard patch-level k-NN scoring and coreset subsampling mechanics utilized in the UAD evaluation pipeline.

---

## Limitations

- **Compute Constraints:** Experiments were bounded by a 100-hour dual-T4 GPU limit, capping coreset scaling at 20,000 vectors.
- **Single-Encoder Comparison:** The study rigorously contrasts I-JEPA ViT-H/14 against DINOv2 ViT-L/14; findings regarding layer-depth degradation may differ in reconstructive (MAE) or contrastive (CLIP) architectures.
- **Pixel-Level Localization:** Constrained by the 14-pixel patch resolution native to the ViT architecture. Gaussian pre-smoothing mitigates this, but true sub-patch localization requires multi-scale aggregation. 
```
