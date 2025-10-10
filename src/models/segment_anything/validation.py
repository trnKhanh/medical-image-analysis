from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as F
import SimpleITK as sitk
from medpy import metric
from scipy.ndimage import zoom
from einops import repeat
from PIL import Image

from utils import get_path, draw_mask, zoom_image


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1

    dice = 0
    hd95 = np.nan

    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)

    return dice, hd95


def test_single_volume(
    image,
    label,
    net,
    classes,
    multimask_output=True,
    patch_size=[512, 512],
    loss_fn=None,
):
    _, _, D, H, W = image.shape
    image = image.to(net.device)  # B, C, D, H, W
    label = label.to(net.device)  # B, D, H, W
    image = image.squeeze(0).permute(1, 0, 2, 3)  # D, C, H, W
    label = label.squeeze(0)  # D, H, W

    resized_image = zoom_image(image, patch_size, order=3)
    resized_label = zoom_image(label, patch_size, order=0)

    net.eval()
    ensemble_output_masks = torch.zeros(1, device=net.device)
    with torch.no_grad():
        outputs = net(resized_image, multimask_output, patch_size[0])
        output_masks = outputs["masks"]
        for m in output_masks:
            ensemble_output_masks = ensemble_output_masks + m.softmax(1)
        prediction = ensemble_output_masks.argmax(1)

        prediction = zoom_image(prediction, [H, W], order=0)
        assert isinstance(prediction, torch.Tensor)

    if loss_fn:
        loss = torch.Tensor(
            [loss_fn(m, resized_label) for m in output_masks]
        ).mean(0)
    else:
        loss = None

    prediction = prediction.cpu().numpy()
    label = label.cpu().numpy()

    metric_list = []
    for i in range(1, classes):
        metric_list.append(
            calculate_metric_percase(prediction == i, label == i)
        )

    return metric_list, loss


def test_single_volume_prompt(
    image,
    label,
    net,
    classes,
    promptidx,
    promptmode,
    multimask_output=True,
    patch_size=[512, 512],
    loss_fn=None,
):
    _, _, D, H, W = image.shape
    image = image.to(net.device)  # B, C, D, H, W
    label = label.to(net.device)  # B, D, H, W
    image = image.squeeze(0).permute(1, 0, 2, 3)  # D, C, H, W
    label = label.squeeze(0)  # D, H, W

    resized_image = zoom_image(image, patch_size, order=3)
    resized_label = zoom_image(label, patch_size, order=0)

    net.eval()
    ensemble_output_masks = torch.zeros(1, device=net.device)
    with torch.no_grad():
        outputs = net(
            resized_image,
            multimask_output,
            patch_size[0],
            promptidx,
            promptmode,
        )
        output_masks = outputs["masks"]
        for m in output_masks:
            ensemble_output_masks = ensemble_output_masks + m.softmax(1)
        prediction = ensemble_output_masks.argmax(1)
        prediction = zoom_image(prediction, [H, W], order=0)
        assert isinstance(prediction, torch.Tensor)

    if loss_fn:
        loss, _, _ = torch.Tensor(
            [loss_fn(m, resized_label) for m in output_masks]
        ).mean(0)
    else:
        loss = None

    prediction = prediction.cpu().numpy()
    label = label.cpu().numpy()

    metric_list = []
    for i in range(1, classes):
        metric_list.append(
            calculate_metric_percase(prediction == i, label == i)
        )

    return metric_list, loss


def calculate_metric_percase_nan(pred, gt, raw_spacing):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() != 0:
        dice = metric.binary.dc(pred, gt)
        asd = metric.binary.asd(pred, gt, raw_spacing)
        hd95 = metric.binary.hd95(pred, gt, raw_spacing)
        jc = metric.binary.jc(pred, gt)
    else:
        dice = 0
        hd95 = np.nan
        asd = np.nan
        jc = 0
    return dice, hd95, asd, jc

def test_single_volume_mean(
    data_path: Path,
    image,
    label,
    net,
    classes,
    multimask_output,
    patch_size=[512, 512],
    input_size=[224, 224],
    test_save_path: Path | None = None,
    case=None,
    z_spacing=1,
):
    assert image.shape[0] == 1
    _, _, D, H, W = image.shape
    image = image.to(net.device)  # B, C, D, H, W
    label = label.to(net.device)  # B, D, H, W
    image = image.squeeze(0).permute(1, 0, 2, 3)  # D, C, H, W
    label = label.squeeze(0)  # D, H, W

    resized_image = zoom_image(image, patch_size, order=3)
    resized_label = zoom_image(label, patch_size, order=0)

    net.eval()
    with torch.no_grad():
        outputs = net(resized_image, multimask_output, patch_size[0])
        output_masks = outputs["masks"]
        ensemble_output_masks = torch.zeros(1, device=net.device)
        for mask in output_masks:
            ensemble_output_masks = ensemble_output_masks + mask.softmax(1)
        prediction = ensemble_output_masks.argmax(1)
        prediction = zoom_image(prediction, [H, W], order=0)
        assert isinstance(prediction, torch.Tensor)

    prediction = prediction.cpu().numpy()
    image = image.cpu().numpy()
    label = label.cpu().numpy()

    # get resolution
    case_raw = data_path / f"ACDC_raw/{case}.nii.gz"
    case_raw = sitk.ReadImage(case_raw)
    raw_spacing = case_raw.GetSpacing()
    raw_spacing_new = []
    raw_spacing_new.append(raw_spacing[2])
    raw_spacing_new.append(raw_spacing[1])
    raw_spacing_new.append(raw_spacing[0])
    raw_spacing = raw_spacing_new

    metric_list = []
    for i in range(1, classes):
        metric_list.append(
            calculate_metric_percase_nan(
                prediction == i, label == i, raw_spacing
            )
        )

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path / f"{case}_pred.nii.gz")
        # sitk.WriteImage(img_itk, test_save_path + '/' + case + "_img.nii.gz")
        # sitk.WriteImage(lab_itk, test_save_path + '/' + case + "_gt.nii.gz")

        result_path = get_path(test_save_path) / str(case)

        label_path = result_path / "label"
        visual_path = result_path / "visual"
        label_path.mkdir(parents=True, exist_ok=True)
        visual_path.mkdir(parents=True, exist_ok=True)

        for i in range(prediction.shape[0]):
            slice = (image[i][0] * 255).astype(np.uint8)
            mask = prediction[i].astype(np.uint8)
            slice_label = label[i].astype(np.uint8)

            mask_pil = Image.fromarray(mask)
            visual = draw_mask(slice, slice_label, 0.2)
            visual = draw_mask(visual, mask, 0.4)
            visual_pil = Image.fromarray(visual)

            mask_pil.save(label_path / f"slice_{i}.png")
            visual_pil.save(visual_path / f"slice_{i}.png")

    return metric_list
