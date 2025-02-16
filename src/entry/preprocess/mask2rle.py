import uuid
import json
from pathlib import Path
from argparse import ArgumentParser

from label_studio_converter.brush import mask2rle
import numpy as np
from PIL import Image


def parse_args():
    parser = ArgumentParser("Convert masks to label studio RLE format")

    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--label-dir")
    parser.add_argument("--unlabel-dir")
    parser.add_argument("--output-path", required=True)

    return parser.parse_args()


def mask2annotation(
    mask,
    label_names,
    from_name,
    to_name,
    ground_truth=False,
    model_version=None,
    score=None,
):
    width, height = mask.shape
    result = {"result": []}
    for class_id in label_names.keys():
        rle = mask2rle((mask == class_id) * 255)
        res = {
            "id": str(uuid.uuid4())[0:8],
            "type": "brushlabels",
            "value": {
                "rle": rle,
                "format": "rle",
                "brushlabels": [label_names[class_id]],
            },
            "origin": "manual",
            "to_name": to_name,
            "from_name": from_name,
            "image_rotation": 0,
            "original_width": width,
            "original_height": height,
        }
        result["result"].append(res)

    # prediction
    if model_version:
        result["model_version"] = model_version
        result["score"] = score

    # annotation
    else:
        result["ground_truth"] = ground_truth

    return result


PREFIX = {"label": "labeled_data_", "unlabel": "unlabeled_data_"}


def mask2rle_entry():
    args = parse_args()

    image_dir_path = Path(args.image_dir)
    label_dir_path = Path(args.label_dir)
    unlabel_dir_path = Path(args.unlabel_dir)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    project_data = []
    for image_path in sorted(image_dir_path.glob("*.png")):
        image_id = image_path.stem
        if PREFIX["unlabel"] in image_id:
            image_number = image_id.replace(PREFIX["unlabel"], "")
            is_labeled = False
        else:
            image_number = image_id.replace(PREFIX["label"], "")
            is_labeled = True

        if is_labeled:
            try:
                mask_path = label_dir_path / f"{image_id}.png"
                mask = np.array(Image.open(mask_path).convert("L"))
            except:
                mask_path = label_dir_path / f"{image_number}.png"
                mask = np.array(Image.open(mask_path).convert("L"))
        else:
            try:
                mask_path = unlabel_dir_path / f"{image_id}.png"
                mask = np.array(Image.open(mask_path).convert("L"))
            except:
                mask_path = unlabel_dir_path / f"{image_number}.png"
                mask = np.array(Image.open(mask_path).convert("L"))

        annotation = {
            "data": {
                "image": f'http://localhost:8001/{str((image_dir_path / f"{image_id}.png"))}',
                "id": image_id,
                "type": "labeled" if is_labeled else "unlabeled",
            },
            "predictions": [
                mask2annotation(
                    mask,
                    {1: "anterior lip", 2: "posterior lip"},
                    "tag",
                    "image",
                )
            ],
        }
        project_data.append(annotation)

    with open(output_path, "w") as f:
        json.dump(project_data, f, indent=2)
