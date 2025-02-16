import json
from collections import deque
from argparse import ArgumentParser
from pathlib import Path

from label_studio_converter.brush import decode_rle
import numpy as np
from PIL import Image

from utils import draw_mask

def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--label-dir", required=True)
    parser.add_argument("--mask-file", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--threshold", type=int, required=True)
    parser.add_argument("--visualize", action="store_true")

    return parser.parse_args()

CLASS_DICT = {
    "anterior lip": 1,
    "posterior lip": 2,
}

def remove_noise(image, threshold):
    h, w = image.shape
    res = image.copy()
    st = []
    def bfs(x, y):
        q = deque()
        q.append((x, y))
        visisted[x, y] = True
        cnt = 0
        while len(q):
            x, y = q.popleft()
            st.append((x, y))
            cnt += 1
            for dx, dy in adj:
                newX = x + dx
                newY = y + dy
                if newX < 0 or newX >= h:
                    continue
                if newY < 0 or newY >= w:
                    continue
                if image[newX, newY] != image[x, y]:
                    continue
                if visisted[newX, newY]:
                    continue
                q.append((newX, newY))
                visisted[newX, newY] = True
        return cnt

    adj = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx != 0 and dy != 0:
                adj.append((dx, dy))

    visisted = np.zeros_like(image, dtype=np.bool_)
    visisted[image == 0] = True

    for x in range(h):
        for y in range(w):
            if visisted[x, y]:
                continue
            st = []
            cnt = bfs(x, y)
            if cnt < threshold:
                for X, Y in st:
                    res[X, Y] = 255 - res[X, Y]
    return res
        

def rle2mask_entry():
    args = parse_args()

    image_dir_path = Path(args.image_dir)
    label_dir_path = Path(args.label_dir)
    mask_file_path = Path(args.mask_file) 
    save_dir_path = Path(args.save_dir) 

    (save_dir_path / "images").mkdir(exist_ok=True, parents=True)
    (save_dir_path / "labels").mkdir(exist_ok=True, parents=True)
    (save_dir_path / "visualized").mkdir(exist_ok=True, parents=True)

    with open(mask_file_path, "r")  as f:
        data = json.load(f)

    for task in data:
        masks = task["annotations"][0]["result"]
        image_size = (masks[0]["original_width"], masks[0]["original_height"], 4)
        image_id = task["data"]["id"]

        final_mask = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)
        mask_dict = {}
        for mask in masks:
            rle = mask["value"]["rle"]
            label = CLASS_DICT[mask["value"]["brushlabels"][0]]
            mask_np = decode_rle(rle).reshape((image_size[1], image_size[0], 4))[:, :, 0]
            mask_np[mask_np > 0] = 255
            denoised_mask = remove_noise(mask_np, args.threshold)
            mask_dict[label] = denoised_mask

        for label in [2, 1]:
            mask_pos = mask_dict[label] > 0
            final_mask[mask_pos] = label

        Image.fromarray(final_mask).save(save_dir_path / "labels" / f"{image_id}.png")

        image_path = image_dir_path / f"{image_id}.png"
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        image.save(save_dir_path / "images" / f"{image_id}.png")

        visualized_image = draw_mask(image_np, final_mask)
        Image.fromarray(visualized_image).save(save_dir_path/ "visualized" / f"{image_id}.png")

    for label_path in label_dir_path.glob("*.png"):
        image_id = label_path.stem
        mask = Image.open(label_path).convert("L")
        mask_np = np.array(mask)
        mask.save(save_dir_path / "labels" / f"labeled_data_{image_id}.png")

        image_path = image_dir_path / f"labeled_data_{image_id}.png"
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        image.save(save_dir_path / "images" / f"labeled_data_{image_id}.png")

        visualized_image = draw_mask(image_np, mask_np)
        Image.fromarray(visualized_image).save(save_dir_path/ "visualized" / f"labeled_data_{image_id}.png")

