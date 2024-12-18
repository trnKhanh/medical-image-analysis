---
Date created: 18-12-2024
Time created: 18:23
---

# Summary
- Name: UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation.
- Year: 2022 (arXiv), 2024 (IEEE TMI).
- Journal: IEEE TMI.
- Link: [arXiv](https://arxiv.org/abs/2212.04497).
- Task: 3D Image Segmentation.
- Modalities: MRI/CT.
- Areas/Organs: various.

## Method
- Architecture: U-Net with transformer encoder (Swin) and CNN decoder. They use Efficient Paired-Attention (EPA) instead of normal attention to efficiently (linear complexity with input size) capture the rich spatial-channel information.
- Loss function: Soft Dice + Cross Entropy.
- Training method: Supervised.

# Experiments

## Datasets
- Synapse (CT): abdomen.
- BTCV (CT): abdomen.
- ACDC (MRI): heart.
- BraTs (MRI): brain.
- MSD-Lung (CT): lung.

# Comments
