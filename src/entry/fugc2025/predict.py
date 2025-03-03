import os
from pathlib import Path
from argparse import ArgumentParser

import cv2
import torch
import torchvision.transforms.functional as F
from PIL import Image
import numpy as np
from tqdm import tqdm
from models.unet import UNet
from utils.utils import draw_mask
from transforms.normalization import ZScoreNormalize


class model:
    def __init__(self, image_size, folds=[0, 1, 2, 3, 4]):
        """
        This constructor is supposed to initialize data members.
        Use triple quotes for function documentation.
        """
        self.dilate_size = 5
        self.erode_size = 5
        self.smooth_kernel = 7

        self.mean = None
        self.std = None
        self.folds = folds
        self.models = [UNet(3, 3).cpu() for _ in self.folds]
        if image_size and len(image_size) < 2:
            image_size *= 2
        self.image_size = image_size

        self.normalization = ZScoreNormalize()

    def load(self, path="./"):
        for i, fold in enumerate(self.folds):
            model_path = os.path.join(path, f"fold_{fold}/checkpoint_best.pth")
            self.models[i].load_state_dict(
                torch.load(model_path, map_location="cpu")["model"]
            )
        return self

    def preprocess(self, X):
        X = X / 255.0
        image = torch.tensor(X, dtype=torch.float32)
        if self.image_size:
            image = F.resize(
                image, self.image_size, F.InterpolationMode.BILINEAR
            )
        image = image.unsqueeze(0)

        return image

    def postprocess(self, P, ori_shape):
        mask = P.squeeze(0).argmax(dim=0)
        if self.image_size:
            mask = F.resize(mask[None], ori_shape, F.InterpolationMode.NEAREST)[
                0
            ]
        mask = mask.detach().numpy()

        pad_size = max(self.dilate_size, self.erode_size)

        # Denoise the object mask
        object_mask = np.zeros_like(mask, dtype=np.uint8)
        object_mask[mask > 0] = 255
        object_mask = self.pad_mask(object_mask, pad_size)
        for _ in range(1):
            denoised_object_mask = self.remove_cc(self.fill_hole(object_mask))
            denoised_object_mask = self.remove_pad(denoised_object_mask, pad_size)
            object_mask = denoised_object_mask.copy()
        final_object_mask = self.smoothen_boundary(object_mask)

        # Denoise the posterior lip in the object mask
        ant_lip_mask = np.zeros_like(mask, dtype=np.uint8)
        ant_lip_mask[mask == 1] = 255
        ant_lip_mask = self.pad_mask(ant_lip_mask, pad_size)
        for _ in range(1):
            denoised_ant_lip_mask = self.remove_cc(self.fill_hole(ant_lip_mask))
            denoised_ant_lip_mask = self.remove_pad(denoised_ant_lip_mask, pad_size)
            ant_lip_mask = denoised_ant_lip_mask.copy()
        final_ant_lip_mask = self.smoothen_boundary(ant_lip_mask)
        final_ant_lip_mask[final_object_mask == 0] = 0

        mask[final_object_mask == 0] = 0
        mask[final_object_mask > 0] = 2
        mask[final_ant_lip_mask > 0] = 1

        return mask

    def fill_hole(self, mask):
        dilated_mask = self.dilate(mask, self.dilate_size)
        eroded_mask = self.erode(dilated_mask, self.erode_size)

        return eroded_mask

    def remove_cc(self, mask):
        eroded_mask = self.erode(mask, self.erode_size)
        dilated_mask = self.dilate(eroded_mask, self.dilate_size)

        return dilated_mask

    def pad_mask(self, mask, pad_size):
        padded_mask = cv2.copyMakeBorder(
            mask,
            pad_size,
            pad_size,
            pad_size,
            pad_size,
            cv2.BORDER_CONSTANT,
            None,
            [0],
        )
        return padded_mask

    def remove_pad(self, mask, pad_size):
        top = pad_size
        bot = mask.shape[0] - pad_size
        left = pad_size
        right = mask.shape[1] - pad_size
        return mask[top:bot, left:right]


    def dilate(self, mask, dilate_size):
        kernel_size = (dilate_size * 2 + 1, dilate_size * 2 + 1)
        element = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        res = cv2.dilate(mask, element)
        return res

    def erode(self, mask, erode_size):
        kernel_size = (erode_size * 2 + 1, erode_size * 2 + 1)
        element = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        res = cv2.erode(mask, element)
        return res

    def smoothen_boundary(self, mask):
        mask = cv2.GaussianBlur(
            mask, (self.smooth_kernel, self.smooth_kernel), 0
        )
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        return mask

    def predict(self, X, no_normalization=True):
        """
        X: numpy array of shape (3,336,544)
        """
        for model in self.models:
            model.eval()

        ori_shape = [X.shape[-2], X.shape[-1]]
        X = self.preprocess(X)

        P = None
        for model in self.models:
            seg = model(X)
            seg = seg.softmax(1)
            P = P + seg if P is not None else seg

        mask = self.postprocess(P, ori_shape)
        return mask

    def save(self, path="./"):
        """
        Save a trained model.
        """
        pass


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--work-dir", default=".", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument(
        "--images",
        required=True,
        type=str,
        help="Path to image or images directory",
    )
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--visualize-dir", type=str)
    parser.add_argument("--run-model", action="store_true")
    parser.add_argument("--image-size", nargs="+", type=int)
    parser.add_argument("--show", action="store_true")

    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--no-normalization", action="store_true")

    return parser.parse_args()


def predict_entry():
    args = parse_args()

    m = model(args.image_size)
    m.load(args.work_dir)

    images_path = Path(args.images)

    if args.output_dir:
        output_dir_path = Path(args.output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
    else:
        output_dir_path = None

    if args.visualize_dir:
        visualize_dir_path = Path(args.visualize_dir)
        visualize_dir_path.mkdir(parents=True, exist_ok=True)
    else:
        visualize_dir_path = None

    images_iter = (
        tqdm(images_path.glob("*.png"))
        if images_path.is_dir()
        else iter([images_path])
    )

    for image_path in images_iter:
        image_pil = Image.open(image_path).convert("RGB")
        image_np = np.array(image_pil)

        if args.run_model:
            image_input = image_np.transpose(2, 0, 1)

            pred: np.ndarray = m.predict(image_input, args.no_normalization)

            if output_dir_path:
                save_path = output_dir_path / image_path.name
                Image.fromarray(pred.astype(np.uint8)).save(save_path)
        elif output_dir_path:
            save_path = output_dir_path / image_path.name
            pred: np.ndarray = np.array(Image.open(save_path))
        else:
            raise ValueError("Either output-dir or run-model must be specified")

        visualized_image = draw_mask(image_np, pred)
        visualized_pil = Image.fromarray(visualized_image.astype(np.uint8))

        if visualize_dir_path:
            visualize_path = visualize_dir_path / image_path.name
            visualized_pil.save(visualize_path)

        if args.show:
            visualized_pil.show()
