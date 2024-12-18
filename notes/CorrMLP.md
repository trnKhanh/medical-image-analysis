---
Date created: 18-12-2024
Time created: 18:15
---

# Summary
- Name: Correlation-aware Coarse-to-fine MLPs for Deformable Medical Image Registration.
- Year: 2024.
- Conference: CVPR 2024.
- Link: [arXiv](https://arxiv.org/abs/2406.00123).
- Task: Deformable Medical Image Registration.
- Modalities: MRI.
- Areas/Organs: brain, heart.

## Method
- There are two inputs to the models, fixed image ($F_f$) and moving image ($F_m$).
- Model:
	- Feature Extraction Encoder: 4 convolutional blocks are used to extract features at 4 scales: $\{(F_f^1,F_m^1),(F_f^2,F_m^2),(F_f^3,F_m^3),(F_f^4,F_m^4)\}$
	- Correlation-aware Coarse-to-fine Registration Decoder: 
		- Similar to U-Net, they feed the features at smallest scale, $(F_f^4,F_m^4)$, to a CMW-MLP block to obtain $\psi_1$, which is then upsampled before being used to warp $F_m^3$. The warped version of $F_m^3$ is used with $F_f^3$ as the input to the first CMW-MLP block of the second stage. The output of the first block and the upsampled $\psi_1$ is fed to the second CMW-MLP block of this stage, whose output is added with the upsampled $\psi_1$ to obtain $\psi_2$. These steps are repeated for another two times to finally obtain $\psi_4$.
		- CMW-MLP block:
			- Input: $F_1$ and $F_2$.
			- 3D correlation layer [(paper)](https://arxiv.org/abs/1909.11966) is used to compute 3D correlation map $C_F$.
			- $F_1$, $F_2$, and $C_F$ are then concatenated and passed through a convolution layer to obtain $F_{corr}$.
			- $F_{corr}$ is normalized using LayerNorm and passed through multi-branches MLP (each branch corresponds to different region split, and gMLP [(paper)](https://arxiv.org/abs/2105.08050) is adopted to process the features within each region). 
			- The outputs from all branches are fused together using GAP, 2-layer MLP, and the softmax function. Then a residual channel attention module [(paper)](https://arxiv.org/abs/2201.02973) is used to highlight crucial features channels.
- Loss functions:
	- Similarity loss: penalize the different between the fixed image and warped moving image.
	- Diffusion regularizer: smoothen $\psi$. This loss basically penelize large gradient of $\psi$.

# Experiments

## Datasets

### Brain image registration
- Train:
	- ADNI (MRI): brain.
	- ABIDE (MRI): brain.
	- ADHD (MRI): brain.
	- IXI (MRI): brain.
- Validation and test:
	- Mindboggle (MRI): brain.
	- Buckner (RMI): brain.

### Cardiac image registration
- ACDC (4D MRI): heart.

# Comments
