# Method
  - Architecture: U-Net with transformer encoder (Swin) and CNN decoder. They use Efficient Paired-Attention (EPA) instead of normal attention to efficiently (linear complexity with input size) capture the rich spatial-channel information.
  - Loss function: Soft Dice + Cross Entropy.
  - Training method: Supervised.

# Experiment
## Datasets
  - Synapse (CT): abdomen.
  - BTCV (CT): abdomen.
  - ACDC (MRI): heart.
  - BraTs (MRI): brain.
  - MSD-Lung (CT): lung.
