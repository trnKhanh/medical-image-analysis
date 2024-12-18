---
Date created: 18-12-2024
Time created: 18:19
---

# Summary
- Name: Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images.
- Year: 2022.
- Conference: BrainLes@MICCAI 2022.
- Link: [arXiv](https://arxiv.org/abs/2201.01266).
- Task: 3D Image Segmentation.
- Modalities: MRI.
- Areas/Organs: brain.

## Method
- Architecture: U-Net with transformer encoder (Swin) and CNN decoder.
- Loss function: Soft Dice.
- Training method: Supervised.

# Experiments

## Datasets
- BraTs 2021 (MRI): abdomen.

## Implementation
- Ensemble 10 models from 2 separate 5-fold training.

# Comments
- For some reasons, the implementation in MONAI do not transform the features vector of the third layer in U-Net (the one whose size is (C*8, W/16, H/16, D/16)).
