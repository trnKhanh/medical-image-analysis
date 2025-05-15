import tomllib
from typing import Final

import torch
from sklearn.model_selection import train_test_split

from datasets.acdc.kaggle_acdc_dataset import KaggleACDCDataset
from training import ActiveLearningTrainer
from transforms.normalization import MinMaxNormalize

with open("configs/development.toml", "rb") as f:
    CONFIG: Final = tomllib.load(f)

def train_entry():
    train_dir = CONFIG["datasets"]["acdc"]["train"]
    normalize = MinMaxNormalize(0.0, 1.0)
    train_data = KaggleACDCDataset.load(train_dir, is_training=True, target_size=(256, 256))
    train_images, train_masks = train_data["images"], train_data["masks"]
    for i in range(len(train_images)):
        normalized = normalize(train_images[i])  # torch tensor, shape: (1, 256, 256)
        normalized = normalized.numpy().transpose(1, 2, 0)  # shape: (256, 256, 1)
        train_images[i] = normalized

    print(len(train_images), len(train_masks))

    X_train, X_val, y_train, y_val = train_test_split(train_images, train_masks, test_size=0.2, random_state=1337)
    trainer = ActiveLearningTrainer(
        X_train,
        y_train,
        X_val,
        y_val,
        CONFIG["train"]["active_learning"]["checkpoints"],
        initial_size=200,
        budget=100,
        rounds=10,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    trainer.run_training()


if __name__ == "__main__":
    train_entry()
