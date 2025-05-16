import os
import tomllib
from typing import Final

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tqdm import tqdm

from datasets.acdc.kaggle_acdc_dataset import KaggleACDCDataset
from models import UNet

with open("configs/development.toml", "rb") as f:
    CONFIG: Final = tomllib.load(f)

# ========== CONFIG ==========
ROUND = 5
TEST_DIR = CONFIG["datasets"]["acdc"]["test"]
TRAIN_DIR = CONFIG["datasets"]["acdc"]["train"]
CHECKPOINT_PATH = f'{CONFIG["train"]["active_learning"]["checkpoints"]}/unet_round_{ROUND}.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = f"{CONFIG["test"]["active_learning"]["output"]}/round_{ROUND}"


def get_num_classes(filepath) -> int:
    data = KaggleACDCDataset.load(filepath, is_training=True, target_size=(256, 256))
    num_classes = np.max(data['masks']) + 1
    return num_classes

# ========== LOAD MODEL ==========
model = UNet(n_channels=1, n_classes=get_num_classes(TRAIN_DIR))
state = torch.load(CHECKPOINT_PATH, weights_only=False)
model.load_state_dict(state["model_state"])
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


def compute_extended_metrics(y_true, y_pred):
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    dice = 2 * np.sum(y_true_flat * y_pred_flat) / (np.sum(y_true_flat) + np.sum(y_pred_flat) + 1e-6)
    jaccard = np.sum(y_true_flat * y_pred_flat) / (
        np.sum(y_true_flat) + np.sum(y_pred_flat) - np.sum(y_true_flat * y_pred_flat) + 1e-6)
    precision = precision_score(y_true_flat, y_pred_flat, average='macro', zero_division=0)
    recall = recall_score(y_true_flat, y_pred_flat, average='macro', zero_division=0)
    accuracy = accuracy_score(y_true_flat, y_pred_flat)

    return dice, jaccard, precision, recall, accuracy


def test_entry() -> None:
    """Test entry point."""
    sum_dice = 0.0
    sum_jaccard = 0.0
    num_slices = 0
    sum_precision = 0.0
    sum_recall = 0.0
    sum_accuracy = 0.0

    for filename in tqdm(sorted(os.listdir(TEST_DIR))):
        if not filename.endswith('.h5'):
            continue

        path = os.path.join(TEST_DIR, filename)
        img, lbl = load_volume(path)

        for slice_idx in range(img.shape[0]):
            img_slice = img[slice_idx]
            lbl_slice = lbl[slice_idx]

            pred_mask = predict(preprocess(img_slice))

            valid_mask = lbl_slice >= 0

            dice, jaccard, precision, recall, accuracy = compute_extended_metrics(
                lbl_slice[valid_mask], pred_mask[valid_mask]
            )
            sum_precision += precision
            sum_recall += recall
            sum_accuracy += accuracy
            sum_dice += dice
            sum_jaccard += jaccard

            num_slices += 1

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

    mean_dice = sum_dice / num_slices
    mean_iou = sum_jaccard / num_slices
    mean_precision = sum_precision / num_slices
    mean_recall = sum_recall / num_slices
    mean_accuracy = sum_accuracy / num_slices

    print(f"Mean Dice Coefficient: {mean_dice:.4f}")
    print(f"Mean IoU Score: {mean_iou:.4f}")
    print(f"Mean Precision: {mean_precision:.4f}")
    print(f"Mean Recall: {mean_recall:.4f}")
    print(f"Mean Accuracy: {mean_accuracy:.4f}")


if __name__ == '__main__':
    test_entry()
