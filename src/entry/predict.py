import os
from pathlib import Path
from argparse import ArgumentParser

import torch
import torchvision.transforms.functional as F
from PIL import Image
import numpy as np
from tqdm import tqdm

from models.unet import UNet
from utils.utils import draw_mask
from transforms.normalization import ZScoreNormalize


class model:
    def __init__(self, image_size):
        """
        This constructor is supposed to initialize data members.
        Use triple quotes for function documentation.
        """
        self.mean = None
        self.std = None
        self.model = [UNet(3, 3).cpu() for _ in range(5)]
        if len(image_size) < 2:
            image_size *= 2
        self.image_size = image_size

        self.normalization = ZScoreNormalize()

    def load(self, path="./"):
        for i in range(5):
            model_path = os.path.join(path, f"fold_{i}/checkpoint_best.pth")
            self.model[i].load_state_dict(
                torch.load(model_path, map_location="cpu")["model"]
            )
        return self

    def predict(self, X, no_normalization=True):
        """
        X: numpy array of shape (3,336,544)
        """
        for i in range(5):
            self.model[i].eval()
        X = X / 255.0
        image = torch.tensor(X, dtype=torch.float32)
        ori_shape = [image.shape[-2], image.shape[-1]]
        image = F.resize(image, self.image_size, F.InterpolationMode.BILINEAR)
        # image, _ = self.normalization(image, None)
        image = image.unsqueeze(0)

        total_prob = None
        for i in range(5):
            seg = self.model[i](image)  # seg (1,3,336,544)
            seg = seg.softmax(1)
            if total_prob is not None:
                total_prob += seg
            else:
                total_prob = seg

        total_prob = (
            total_prob.squeeze(0).argmax(dim=0)
        )  # (336,544) values:{0,1,2} 1 upper 2 lower
        total_prob = F.resize(total_prob[None], ori_shape, F.InterpolationMode.NEAREST)[0]
        total_prob = total_prob.detach().numpy()

        return total_prob

    def save(self, path="./"):
        """
        Save a trained model.
        """
        pass


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--work-dir", default=".", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--images-dir", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--visualize-dir", required=True, type=str)
    parser.add_argument("--run-model", action="store_true")
    parser.add_argument("--image-size", nargs="+", type=int)

    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--no-normalization", action="store_true")

    return parser.parse_args()


def predict_entry():
    args = parse_args()

    m = model(args.image_size)
    m.load(args.work_dir)

    images_dir_path = Path(args.images_dir)
    output_dir_path = Path(args.output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    if args.visualize_dir:
        visualize_dir_path = Path(args.visualize_dir)
        visualize_dir_path.mkdir(parents=True, exist_ok=True)
    else:
        visualize_dir_path = None

    for image_path in tqdm(images_dir_path.glob("*.png")):
        image_pil = Image.open(image_path).convert("RGB")
        image_np = np.array(image_pil)
        save_path = output_dir_path / image_path.name
        if args.run_model:
            image_input = image_np.transpose(2, 0, 1)

            pred: np.ndarray = m.predict(image_input, args.no_normalization)
            Image.fromarray(pred.astype(np.uint8)).save(save_path)
        else:
            pred: np.ndarray = np.array(Image.open(save_path))

        if visualize_dir_path:
            visualized_image = draw_mask(image_np, pred)
            visualize_path = visualize_dir_path / image_path.name
            Image.fromarray(visualized_image.astype(np.uint8)).save(
                visualize_path
            )
