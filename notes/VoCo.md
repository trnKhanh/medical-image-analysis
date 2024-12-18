---
Date created: 18-12-2024
Time created: 18:26
---

# Summary
- Name: VoCo: A Simple-yet-Effective Volume Contrastive Learning Framework for 3D Medical Image Analysis.
- Year: 2024.
- Conference: CVPR 2024.
- Link: [arXiv](https://arxiv.org/abs/2402.17300).
- Task: 3D Image Segmentation.
- Modalities: CT, MRI.
- Areas/Organs: various.

## Method
- Backbones: Swin-UNETR, 3D U-Net.
- Self-supervised contrastive learning:
	- The 3D volume is divided into base patches (similar to ViT) along the X and Y axis.
	- A patch is randomly cropped. 
	- A backbone with multiple projector heads is used to map these patches to different vector space. They use two different branch:
		- Predictor branch: the branch is used to predict the overlap probability of the random-cropped patch with the base patches using Cosine Similarity. Note that the gradient is only back-propagated to path of the random-cropped patch.
		- Regularizer branch: the feature vectors of the base patches repel each other (contrastive).
- Loss function: 
	- Predictor: negative log of L1 distance between the ground truth labels and prediction logits.
	- Regularizer: cosine similarity between every pair of base patches.

# Experiments

## Datasets

### Pre-train
- BTCV (CT): abdomen.
- TCIA Covid19 (CT): lung.
- LUNA (CT): lung.

### Downstream
- BTCV (CT): abdomen.
- LiTs (CT): liver.
- MSD-Spleen (CT): spleen.
- MM-WHS (CT): heart.
- BraTs (MRI): brain.
- CC-CCI (CT): chest.

# Comments
- Their proposed SSL method aims at capture the contextual position of different patches in the images, which is different from the absolute position.
- The loss functions that they use is relatively strange. Could Cross Entropy be used instead of L1 distance for predictor branch?
- Their implementation simply concatenates all the vectors from all encoding scales and transform them using projector heads.
