from pathlib import Path

import numpy as np
import torch
import SimpleITK as sitk
from medpy import metric
from scipy.ndimage import zoom
from einops import repeat


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    else:
        return 0, np.inf


def test_single_volume(
    image,
    label,
    net,
    classes,
    multimask_output=True,
    patch_size=[512, 512],
    loss_fn=None,
):
    image, label = (
        image.squeeze(0).cpu().detach().numpy(),
        label.squeeze(0).cpu().detach().numpy(),
    )
    prediction = np.zeros_like(label)
    loss_list = []

    for ind in range(image.shape[0]):

        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)
        input = (
            torch.from_numpy(slice)
            .unsqueeze(0)
            .unsqueeze(0)
            .float()
            .to(net.device)
        )
        inputs = repeat(input, "b c h w -> b (repeat c) h w", repeat=3)

        net.eval()

        with torch.no_grad():
            outputs = net(inputs, multimask_output, patch_size[0])
            output_masks = outputs["masks"]

            if loss_fn:
                label_slice = zoom(label[ind], (patch_size[0] / x, patch_size[1] / y), order=0)
                loss_list.append(
                    loss_fn(
                        output_masks, torch.from_numpy(label_slice).unsqueeze(0).long().to(net.device)
                    )
                )

            out = torch.argmax(
                torch.softmax(output_masks, dim=1), dim=1
            ).squeeze(0)
            out = out.cpu().detach().numpy()
            out_h, out_w = out.shape
            if x != out_h or y != out_w:
                pred = zoom(out, (x / out_h, y / out_w), order=0)
            else:
                pred = out
            prediction[ind] = pred

    metric_list = []
    for i in range(1, classes):
        metric_list.append(
            calculate_metric_percase(prediction == i, label == i)
        )
    if len(loss_list):
        losses = torch.Tensor(loss_list)
        loss = losses.mean()
    else:
        loss = None

    return metric_list, loss


def test_single_image(
    image, label, net, classes, multimask_output=True, patch_size=[512, 512]
):

    image, label = (
        image.squeeze(0).cpu().detach().numpy(),
        label.squeeze(0).cpu().detach().numpy(),
    )
    prediction = np.zeros_like(label)

    # for ind in range(image.shape[0]):

    slice = image
    x, y = slice.shape[0], slice.shape[1]
    slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)
    input = (
        torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().to(net.device)
    )
    inputs = repeat(input, "b c h w -> b (repeat c) h w", repeat=3)

    net.eval()

    with torch.no_grad():
        outputs = net(inputs, multimask_output, patch_size[0])
        output_masks = outputs["masks"]
        out = torch.argmax(torch.softmax(output_masks, dim=1), dim=1).squeeze(0)
        out = out.cpu().detach().numpy()
        out_h, out_w = out.shape
        if x != out_h or y != out_w:
            pred = zoom(out, (x / out_h, y / out_w), order=0)
        else:
            pred = out
        prediction = pred

    metric_list = []
    for i in range(1, classes):
        metric_list.append(
            calculate_metric_percase(prediction == i, label == i)
        )
    return metric_list


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

    image, label = (
        image.squeeze(0).cpu().detach().numpy(),
        label.squeeze(0).cpu().detach().numpy(),
    )
    prediction = np.zeros_like(label)
    loss_list = []

    for ind in range(image.shape[0]):

        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)
        input = (
            torch.from_numpy(slice)
            .unsqueeze(0)
            .unsqueeze(0)
            .float()
            .to(net.device)
        )
        inputs = repeat(input, "b c h w -> b (repeat c) h w", repeat=3)

        net.eval()

        with torch.no_grad():
            outputs = net(
                inputs, multimask_output, patch_size[0], promptidx, promptmode
            )
            output_masks = outputs["masks"]

            if loss_fn:
                label_slice = zoom(label[ind], (patch_size[0] / x, patch_size[1] / y), order=0)
                loss_list.append(
                    loss_fn(
                        output_masks, torch.from_numpy(label_slice).unsqueeze(0).long().to(net.device)
                    )
                )

            out = torch.argmax(
                torch.softmax(output_masks, dim=1), dim=1
            ).squeeze(0)
            out = out.cpu().detach().numpy()
            out_h, out_w = out.shape
            if x != out_h or y != out_w:
                pred = zoom(out, (x / out_h, y / out_w), order=0)
            else:
                pred = out
            prediction[ind] = pred

    metric_list = []
    for i in range(1, classes):
        metric_list.append(
            calculate_metric_percase(prediction == i, label == i)
        )

    if len(loss_list):
        losses = torch.Tensor(loss_list)
        loss = losses.mean()
    else:
        loss = None

    return metric_list, loss


def test_single_volume_scm(
    image,
    label,
    net1,
    net2,
    scm,
    classes,
    multimask_output=True,
    patch_size=[512, 512],
):

    image, label = (
        image.squeeze(0).cpu().detach().numpy(),
        label.squeeze(0).cpu().detach().numpy(),
    )
    prediction = np.zeros_like(label)

    for ind in range(image.shape[0]):

        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)
        input = (
            torch.from_numpy(slice)
            .unsqueeze(0)
            .unsqueeze(0)
            .float()
            .to(net1.device)
        )
        inputs = repeat(input, "b c h w -> b (repeat c) h w", repeat=3)

        net1.eval()
        net2.eval()
        scm.eval()

        with torch.no_grad():

            outputs1 = net1(inputs, multimask_output, patch_size[0])
            output_masks1 = outputs1["masks"]
            output1 = torch.softmax(output_masks1, dim=1)

            outputs2 = net2(inputs, multimask_output, patch_size[0])
            output_masks2 = outputs2["masks"]
            output2 = torch.softmax(output_masks2, dim=1)

            conv_in = torch.cat([output1, output2], dim=1)
            conv_out = scm(conv_in)

            out = torch.argmax(torch.softmax(conv_out, dim=1), dim=1).squeeze(0)

            out = out.cpu().detach().numpy()

            out_h, out_w = out.shape
            if x != out_h or y != out_w:
                pred = zoom(out, (x / out_h, y / out_w), order=0)
            else:
                pred = out
            prediction[ind] = pred

    metric_list = []
    for i in range(1, classes):
        metric_list.append(
            calculate_metric_percase(prediction == i, label == i)
        )
    return metric_list


def calculate_metric_percase_nan(pred, gt, raw_spacing):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    if pred.sum() != 0:
        asd = metric.binary.asd(pred, gt, raw_spacing)
        hd95 = metric.binary.hd95(pred, gt, raw_spacing)
    else:
        asd = np.nan
        hd95 = np.nan
    jc = metric.binary.jc(pred, gt)
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
    assert len(image.shape) == 3

    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        if x != patch_size[0] or y != patch_size[1]:
            slice = zoom(
                slice, (patch_size[0] / x, patch_size[1] / y), order=3
            )  # previous using 0, patch_size[0], patch_size[1]
        inputs = (
            torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        )
        inputs = repeat(inputs, "b c h w -> b (repeat c) h w", repeat=3)
        net.eval()
        with torch.no_grad():
            outputs = net(inputs, multimask_output, patch_size[0])
            output_masks1 = outputs["masks"]
            output_masks2 = outputs["masks2"]
            output_masks1 = torch.softmax(output_masks1, dim=1)
            output_masks2 = torch.softmax(output_masks2, dim=1)
            output_masks = (output_masks1 + output_masks2) / 2.0
            out = torch.argmax(
                torch.softmax(output_masks, dim=1), dim=1
            ).squeeze(0)
            out = out.cpu().detach().numpy()
            out_h, out_w = out.shape
            if x != out_h or y != out_w:
                pred = zoom(out, (x / out_h, y / out_w), order=0)
            else:
                pred = out
            prediction[ind] = pred
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
    for i in range(1, classes + 1):
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
    return metric_list
