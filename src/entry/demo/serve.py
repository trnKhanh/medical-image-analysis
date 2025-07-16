import shutil
import uuid
import zipfile
from copy import deepcopy
from functools import partial
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import torchvision.transforms.functional as F
from open_clip import create_model_from_pretrained, get_tokenizer
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader

from activelearning import KMeanSelector
from datasets import ActiveDataset, ExtendableDataset, ImageDataset
from models.unet import UNet, UnetProcessor
from utils import draw_mask

IMAGES_PER_ROW = 10
IMAGE_SIZE = 256
ROOT_DIR = Path(".")
DATA_DIR = ROOT_DIR / "data"

train_set = []
pool_set = []
current_dataset = "dataset"
feature_dict = None


class Config:
    def __init__(self):
        self.budget = 10
        self.model = "BiomedCLIP"
        self.device = torch.device("cpu")
        self.batch_size = 4
        self.loaded_feature_weight = 1
        self.sharp_factor = 1
        self.loaded_feature_only = False
        self.model_ckpt = "./init_model.pth"


config = Config()


def build_foundation_model(device):
    if config.model == "BiomedCLIP":
        model, preprocess = create_model_from_pretrained(
            "hf-hub:microsoft/biomedclip-pubmedbert_256-vit_base_patch16_224"
        )
        tokenizer = get_tokenizer("hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
        model.to(device)
        model.eval()
        return model, preprocess
    else:
        raise RuntimeError()


def build_specialist_model():
    model = UNet(
        dimension=2,
        input_channels=1,
        output_classes=3,
        channels_list=[32, 64, 128, 256, 512],
        block_type="plain",
        normalization="batch",
    )
    model_processor = UnetProcessor(image_size=(IMAGE_SIZE, IMAGE_SIZE))
    return model, model_processor


specialist_model, specialist_processor = build_specialist_model()


def load_specialist_model(model_ckpt):
    specialist_model.load_state_dict(torch.load(model_ckpt, map_location=torch.device("cpu"), weights_only=True))


def get_feature_dict(batch_size, device, active_dataset: ActiveDataset):
    dataset = ConcatDataset([active_dataset.get_train_dataset(), active_dataset.get_pool_dataset()])
    dataloader = DataLoader(dataset, batch_size=batch_size)

    model, preprocess = build_foundation_model(device)
    feature_dict = {}

    for sampled_batch in dataloader:
        image_batch = sampled_batch["image"]
        image_list = []
        for image in image_batch:
            image_pil = F.to_pil_image(image).convert("RGB")
            image_list.append(preprocess(image_pil))
        image_batch = torch.stack(image_list, dim=0)
        image_batch = image_batch.to(device)

        with torch.no_grad():
            feature_batch = model.encode_image(image_batch)

        for i in range(len(feature_batch)):
            case_name = sampled_batch["case_name"][i]
            feature_dict[case_name] = feature_batch[i]

    return feature_dict


def active_select(
    train_set,
    pool_set,
    budget,
    model_ckpt,
    batch_size,
    device,
    loaded_feature_weight,
    sharp_factor,
    loaded_feature_only,
):
    global feature_dict
    train_dataset = ExtendableDataset(ImageDataset(train_set, image_channels=1, image_size=IMAGE_SIZE))
    pool_dataset = ExtendableDataset(ImageDataset(pool_set, image_channels=1, image_size=IMAGE_SIZE))
    active_dataset = ActiveDataset(train_dataset, pool_dataset)
    if feature_dict is None:
        feature_dict = get_feature_dict(batch_size, device, active_dataset)

    active_selector = KMeanSelector(
        batch_size=4,
        num_workers=1,
        pin_memory=True,
        metric="l2",
        feature_dict=feature_dict,
        loaded_feature_weight=loaded_feature_weight,
        sharp_factor=sharp_factor,
        loaded_feature_only=loaded_feature_only,
    )
    load_specialist_model(model_ckpt)
    return active_selector.select_next_batch(active_dataset, budget, specialist_model, device)


def build_input_ui():
    with gr.Accordion("Input") as blk:
        with gr.Row():
            train_gallery = gr.Gallery(
                label="Train set", allow_preview=False, columns=IMAGES_PER_ROW // 2, show_label=True
            )
            pool_gallery = gr.Gallery(
                label="Pool set", allow_preview=False, columns=IMAGES_PER_ROW // 2, show_label=True
            )

        def gallery_change(image_list, target_set=None):
            global feature_dict
            if image_list is None:
                return

            if target_set == "train":
                global train_set
                train_set = [x[0] for x in image_list]
                feature_dict = None
            elif target_set == "pool":
                global pool_set
                pool_set = [x[0] for x in image_list]
                feature_dict = None

        train_gallery.change(partial(gallery_change, target_set="train"), train_gallery, None)
        pool_gallery.change(partial(gallery_change, target_set="pool"), pool_gallery, None)

        return blk


def build_parameters_ui():
    with gr.Accordion() as blk:
        budget_input = gr.Number(config.budget, label="Budget")
        model_ckpt_input = gr.Text(config.model_ckpt, label="Specialist Model Checkpoint")
        device_input = gr.Dropdown(choices=["cuda", "cpu"], value="cpu", label="Device", interactive=True)
        batch_size_input = gr.Number(config.batch_size, label="Batch Size")
        foundation_model_weight_input = gr.Number(config.loaded_feature_weight, label="foundation_model_weight")
        sharp_factor_input = gr.Number(config.sharp_factor, label="sharp_factor")

        def budget_input_change(x):
            config.budget = int(x)

        budget_input.change(budget_input_change, budget_input, None)

        def model_ckpt_input_change(x):
            config.model_ckpt = x

        model_ckpt_input.change(model_ckpt_input_change, model_ckpt_input, None)

        def device_input_change(x):
            config.device = torch.device(x)

        device_input.change(device_input_change, device_input, None)

        def batch_size_input_change(x):
            config.batch_size = int(x)

        batch_size_input.change(batch_size_input_change, batch_size_input, None)

        def foundation_model_weight_input_change(x):
            config.loaded_feature_weight = x

        foundation_model_weight_input.change(foundation_model_weight_input_change, foundation_model_weight_input, None)

        def sharp_factor_input_change(x):
            config.sharp_factor = x

        sharp_factor_input.change(sharp_factor_input_change, sharp_factor_input, None)
        return blk


class_color_map = {
    1: "#ff0000",
    2: "#00ff00",
}
selected_image = None
selected_set = []
annotated_set = []


def predict_pseudo_label(image_pil):
    image = F.to_tensor(image_pil)
    image = image.unsqueeze(0)
    _, _, H, W = image.shape
    image = specialist_processor.preprocess(image)
    with torch.no_grad():
        pred = specialist_model(image)
        pseudo_label = pred.argmax(1)
    pseudo_label = specialist_processor.postprocess(pseudo_label, [H, W])

    return pseudo_label[0]


def hex_to_rgb(h):
    h = h[1:]
    return [int(h[i : i + 2], 16) for i in range(0, 6, 2)]


def build_active_selection_ui():
    with gr.Accordion("Active Selection") as blk:
        select_button = gr.Button("Select")

        with gr.Row():
            selected_gallary = gr.Gallery(
                label="Selected samples", allow_preview=False, columns=IMAGES_PER_ROW // 2, show_label=True
            )
            annotated_gallary = gr.Gallery(
                label="Annotated samples",
                allow_preview=True,
                columns=IMAGES_PER_ROW // 2,
                show_label=True,
                interactive=False,
            )

        image_editor = gr.ImageEditor(
            label="Image Editor",
            interactive=True,
            sources=(),
            brush=gr.Brush(colors=[c for c in class_color_map.values()], color_mode="fixed"),
            layers=False,
        )
        accept_button = gr.Button("Accept")

        download_button = gr.DownloadButton(label="Download Annotated Dataset", visible=False)

        def select_button_click():
            global selected_set, current_dataset, train_set, pool_set, config, annotated_set
            annotated_samples = [x["path"] for x in annotated_set]
            selected_set = active_select(
                list(set(train_set + annotated_samples)),
                pool_set,
                config.budget,
                config.model_ckpt,
                config.batch_size,
                config.device,
                config.loaded_feature_weight,
                config.sharp_factor,
                config.loaded_feature_only,
            )
            current_dataset = uuid.uuid4()
            return selected_set

        select_button.click(select_button_click, None, selected_gallary)


        def get_editor_value(image_path):
            image_pil = Image.open(image_path).convert("L")
            background = np.array(image_pil.convert("RGBA"))
            pseudo_label = predict_pseudo_label(image_pil).cpu().numpy()
            layer = np.zeros_like(background)
            for cl, color in class_color_map.items():
                bin_mask = pseudo_label == cl
                layer[bin_mask] = hex_to_rgb(color) + [255]

            return {"background": background, "layers": [layer], "composite": None}

        def gallery_select(data: gr.SelectData):
            global selected_image
            selected_image = {
                "index": data.index,
                "path": data.value["image"]["path"],
            }
            return get_editor_value(selected_image["path"])

        selected_gallary.select(gallery_select, None, image_editor)

        def accept_button_click(value):
            global selected_set, selected_image, annotated_set
            if len(value["layers"]) and selected_image:
                layer_np = value["layers"][0]
                binary_layer_np = np.zeros_like(layer_np)
                binary_layer_np[layer_np > 127] = 255
                H, W, _ = layer_np.shape
                mask_np = np.zeros((H, W), dtype=np.uint8)
                for cl, color in class_color_map.items():
                    color_rgb = hex_to_rgb(color)
                    bin_mask = np.all(binary_layer_np[:, :, :3] == color_rgb, axis=-1)
                    mask_np[bin_mask] = cl

                selected_image["image"] = value["background"]
                selected_image["mask"] = mask_np
                image_pil = F.to_pil_image(value["background"]).convert("RGB")
                selected_image["visual"] = draw_mask(image_pil, mask_np)

                selected_set = [deepcopy(x) for x in selected_set if x != selected_image["path"]]
                annotated_set.append(deepcopy(selected_image))
                new_index = min(selected_image["index"], len(selected_set) - 1)
                if new_index >= 0:
                    selected_image = {"index": new_index, "path": selected_set[new_index]}
                    image_editor = get_editor_value(selected_image["path"])
                else:
                    selected_image = None
                    image_editor = None
            else:
                image_editor = None

            _download_button = gr.DownloadButton(value=create_download_dataset(), visible=True)
            return image_editor, selected_set, [x["visual"] for x in annotated_set], _download_button

        accept_button.click(
            accept_button_click, image_editor, [image_editor, selected_gallary, annotated_gallary, download_button]
        )

        return blk


def create_download_dataset():
    dataset_dir = DATA_DIR / "dataset"
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(exist_ok=True, parents=True)

    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"

    images_dir.mkdir(exist_ok=True, parents=True)
    labels_dir.mkdir(exist_ok=True, parents=True)

    zip_file = DATA_DIR / "dataset.zip"

    with zipfile.ZipFile(zip_file, "w") as archive:
        for sample in annotated_set:
            case_name = Path(sample["path"]).stem
            image_np = sample["image"]
            label_np = sample["mask"]

            image_pil = Image.fromarray(image_np)
            label_pil = Image.fromarray(label_np)

            image_pil.save(images_dir / f"{case_name}.png")
            label_pil.save(labels_dir / f"{case_name}.png")

            archive.write(images_dir / f"{case_name}.png", arcname=f"images/{case_name}.png")
            archive.write(labels_dir / f"{case_name}.png", arcname=f"labels/{case_name}.png")

    return zip_file


def serve_entry():
    with gr.Blocks() as demo:
        input_ui = build_input_ui()
        parameters_ui = build_parameters_ui()
        active_selection_ui = build_active_selection_ui()
        demo.launch(inbrowser=True)
