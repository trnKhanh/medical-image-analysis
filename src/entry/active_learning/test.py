import os
import tomllib
from typing import Final

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from datasets.acdc.kaggle_acdc_dataset import KaggleACDCDataset
from models import UNet

with open("configs/development.toml", "rb") as f:
    CONFIG: Final = tomllib.load(f)

# ========== CONFIG ==========
ROUND = 5
TEST_DIR = CONFIG["datasets"]["acdc"]["test"]
CHECKPOINT_PATH = f'{CONFIG["train"]["active_learning"]["checkpoints"]}/unet_round_{ROUND}.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = f"{CONFIG["test"]["active_learning"]["output"]}/round_{ROUND}"


# ========== LOAD MODEL ==========
model = UNet(n_channels=1)
model.init_head(4)
model.load_state_dict(torch.load(CHECKPOINT_PATH))
model.to(DEVICE)
model.eval()

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


# ========== PROCESS & PREDICT ==========
def load_volume(filepath):
    with h5py.File(filepath, 'r') as f:
        image = f['image'][:]
        label = f['label'][:]
    return image, label


def preprocess(image):
    image = image.astype(np.float32)
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)  # Normalize to [0, 1]

    if image.ndim == 2:
        # (H, W) -> (H, W, 1)
        image = np.expand_dims(image, axis=-1)

    # (C, H, W) -> (1, C, H, W)
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(DEVICE)

def predict(image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        prediction = output.argmax(dim=1).squeeze().cpu().numpy()
    return prediction


def dice_coefficient(pred, target, valid_mask):
    pred = pred[valid_mask]
    target = target[valid_mask]

    intersection = (pred == target).sum()
    return (2. * intersection) / (pred.size + target.size)

def iou_score(pred, target, valid_mask):
    pred = pred[valid_mask]
    target = target[valid_mask]

    intersection = (pred == target).sum()
    union = pred.size + target.size - intersection
    return intersection / union if union != 0 else 0

def test_entry() -> None:
    """Test entry point."""
    total_correct = 0
    total_pixels = 0
    total_dice = 0.0
    total_iou = 0.0
    num_slices = 0

    i = 0
    for filename in tqdm(sorted(os.listdir(TEST_DIR))):
        if not filename.endswith('.h5'):
            continue
        i += 1
        if i > 10:
            break

        path = os.path.join(TEST_DIR, filename)
        img, lbl = load_volume(path)

        for slice_idx in range(img.shape[0]):
            img_slice = img[slice_idx]
            lbl_slice = lbl[slice_idx]

            pred_mask = predict(preprocess(img_slice))

            valid_mask = lbl_slice >= 0
            correct = (pred_mask == lbl_slice)[valid_mask].sum()
            total = valid_mask.sum()

            total_correct += correct
            total_pixels += total

            total_dice += dice_coefficient(pred_mask, lbl_slice, valid_mask)
            total_iou += iou_score(pred_mask, lbl_slice, valid_mask)
            num_slices += 1

            # Visualization
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(img_slice, cmap='gray')
            axes[0].set_title('Input Image')
            axes[1].imshow(lbl_slice, cmap='jet')
            axes[1].set_title('Ground Truth')
            axes[2].imshow(pred_mask, cmap='jet')
            axes[2].set_title('Prediction')
            plt.suptitle(filename + f' | Slice {slice_idx}')
            plt.tight_layout()
            save_path = os.path.join(OUTPUT_DIR, f"{filename}_slice_{slice_idx}.png")
            plt.savefig(save_path)
            plt.close()

    accuracy = total_correct / total_pixels
    mean_dice = total_dice / num_slices
    mean_iou = total_iou / num_slices

    print(f"\nOverall pixel accuracy: {accuracy:.4f}")
    print(f"Mean Dice Coefficient: {mean_dice:.4f}")
    print(f"Mean IoU Score: {mean_iou:.4f}")

if __name__ == '__main__':
    test_entry()