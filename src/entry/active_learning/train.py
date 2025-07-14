import tomllib
from argparse import ArgumentParser
from typing import Final

import torch
from sklearn.model_selection import train_test_split

from datasets.acdc.kaggle_acdc_dataset import KaggleACDCDataset
from training import ActiveLearningTrainer
from transforms.normalization import MinMaxNormalize

with open("configs/development.toml", "rb") as f:
    CONFIG: Final = tomllib.load(f)

def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--work-path", default=".", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=1337, type=int)

    # >>> Model parameters
    parser.add_argument("--num-classes", default=3, type=int)
    parser.add_argument("--patch-size", default=512, nargs="+", type=int)
    parser.add_argument("--image-size", default=512, nargs="+", type=int)
    parser.add_argument("--model-ckpt", required=True, type=str)
    # <<< Model parameters

    # >>> Data parameters
    parser.add_argument("--data-path", required=True, type=str)
    parser.add_argument("--labeled-num", default=1, type=int)
    parser.add_argument("--do-augment", action="store_true")
    parser.add_argument("--do-normalize", action="store_true")
    parser.add_argument("--batch-size", default=12, type=int)
    parser.add_argument("--labeled-batch-ratio", default=0.5, type=float)
    parser.add_argument("--num-workers", default=1, type=int)
    parser.add_argument("--pin-memory", action="store_true")
    # <<< Data parameters

    # >>> Training parameters
    parser.add_argument("--optimizer", default="adam", type=str)
    parser.add_argument("--num-epochs", default=10000, type=int)
    parser.add_argument("--num-cycles", default=100, type=int)
    parser.add_argument("--budget", default=10, type=int)
    parser.add_argument("--initial-size", default=200, type=int)
    parser.add_argument("--min-iter", default=10000, type=int)
    parser.add_argument("--warmup-iter", default=5000, type=int)
    parser.add_argument("--start-lr", default=1e-3, type=int)
    parser.add_argument("--lr-scheduler", default="poly", type=str)
    parser.add_argument("--lr-warmup-iter", default=250, type=int)
    parser.add_argument("--save-freq-epoch", default=100, type=int)
    parser.add_argument("--valid-freq-iter", default=200, type=int)
    parser.add_argument("--save-metric", default="dice", type=str)
    parser.add_argument("--dice-weight", default=0.8, type=float)
    parser.add_argument("--query-strategy", default="entropy", type=str)
    parser.add_argument("--early-stop-max-patience", default=None, type=int)
    # <<< Training parameters

    # >>> Log parameters
    parser.add_argument("--quiet", dest="verbose", action="store_false")
    parser.add_argument("--log-path", default=None, type=str)
    parser.add_argument("--config-path", default=None, type=str)
    parser.add_argument("--exp-name", default="", type=str)
    # <<< Log parameters

    return parser.parse_args()

def train_entry():
    args = parse_args()
    args_dict = vars(args)
    optimizer = args_dict.pop("optimizer")
    lr_scheduler = args_dict.pop("lr_scheduler")
    save_metric = args_dict.pop("save_metric")
    trainer = ActiveLearningTrainer(
        optimizer_name=optimizer,
        optimizer_kwargs={},
        lr_scheduler_name=lr_scheduler,
        save_metric_name=save_metric,
        **args_dict,
    )
    trainer.initialize()
    trainer.run_training()

# def train_entry():
#     train_dir = CONFIG["datasets"]["acdc"]["train"]
#     normalize = MinMaxNormalize(0.0, 1.0)
#     train_data = KaggleACDCDataset.load(train_dir, is_training=True, target_size=(256, 256))
#     train_images, train_masks = train_data["images"], train_data["masks"]
#     for i in range(len(train_images)):
#         normalized = normalize(train_images[i])  # torch tensor, shape: (1, 256, 256)
#         normalized = normalized.numpy().transpose(1, 2, 0)  # shape: (256, 256, 1)
#         train_images[i] = normalized
#
#     print(len(train_images), len(train_masks))
#
#     X_train, X_val, y_train, y_val = train_test_split(train_images, train_masks, test_size=0.2, random_state=1337)
#     trainer = ActiveLearningTrainer(
#         X_train,
#         y_train,
#         X_val,
#         y_val,
#         CONFIG["train"]["active_learning"]["checkpoints"],
#         initial_size=200,
#         budget=100,
#         rounds=10,
#         device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     )
#     trainer.run_training()


if __name__ == "__main__":
    train_entry()
