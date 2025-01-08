import SimpleITK as sitk

import torch
from torch import nn

import numpy as np


class HD(nn.Module):
    def __init__(self):
        super(HD, self).__init__()
        pass

    def numpy_to_image(self, image) -> sitk.Image:
        image = sitk.GetImageFromArray(image)
        return image

    def evaluation(self, pred: sitk.Image, label: sitk.Image):
        result = dict()

        # 计算upper指标
        pred_data_upper = sitk.GetArrayFromImage(pred)
        pred_data_upper[pred_data_upper == 2] = 0
        # pred_upper = sitk.GetImageFromArray(pred_data_upper)

        label_data_upper = sitk.GetArrayFromImage(label)
        label_data_upper[label_data_upper == 2] = 0
        # label_upper = sitk.GetImageFromArray(label_data_upper)

        result["hd_upper"] = float(self.cal_hd(pred_data_upper, label_data_upper))

        # 计算lower指标
        pred_data_lower = sitk.GetArrayFromImage(pred)
        pred_data_lower[pred_data_lower == 1] = 0
        pred_data_lower[pred_data_lower == 2] = 1
        # pred_lower = sitk.GetImageFromArray(pred_data_lower)

        label_data_lower = sitk.GetArrayFromImage(label)
        label_data_lower[label_data_lower == 1] = 0
        label_data_lower[label_data_lower == 2] = 1
        # label_lower = sitk.GetImageFromArray(label_data_lower)

        result["hd_lower"] = float(self.cal_hd(pred_data_lower, label_data_lower))

        # 计算总体指标
        pred_data_all = sitk.GetArrayFromImage(pred)
        pred_data_all[pred_data_all == 2] = 1
        # pred_all = sitk.GetImageFromArray(pred_data_all)

        label_data_all = sitk.GetArrayFromImage(label)
        label_data_all[label_data_all == 2] = 1
        # label_all = sitk.GetImageFromArray(label_data_all)

        result["hd_all"] = float(self.cal_hd(pred_data_all, label_data_all))

        return (result["hd_all"] + result["hd_lower"] + result["hd_upper"]) / 3

    def forward(self, pred, label):
        """
        :param pred: (BS,3,336,544)
        :param label: (BS,336,544)
        :return:
        """
        # print(pred.shape)
        # print(label.shape)

        pred = (
            torch.argmax(pred, dim=1)[0].detach().cpu().numpy().astype(np.int64)
        )  # (H,W) value:0,1,2  1-upper 2-lower
        label = (
            label[0].detach().cpu().numpy().astype(np.int64)
        )  # (H,W) value:0,1,2  1-upper 2-lower
        pre_image = self.numpy_to_image(pred)
        truth_image = self.numpy_to_image(label)
        result = self.evaluation(pre_image, truth_image)

        return result

    def cal_hd(self, a, b):
        try:
            sum_a = np.sum(a)
            sum_b = np.sum(b)
            if sum_a == 0 and sum_b == 0:
                return 0.0
            elif sum_a == 0 or sum_b == 0:
                return np.inf

            a = sitk.GetImageFromArray(a)
            b = sitk.GetImageFromArray(b)

            a = sitk.Cast(sitk.RescaleIntensity(a), sitk.sitkUInt8)
            b = sitk.Cast(sitk.RescaleIntensity(b), sitk.sitkUInt8)
            filter1 = sitk.HausdorffDistanceImageFilter()
            filter1.Execute(a, b)
            hd = filter1.GetHausdorffDistance()
            return hd
        except:
            return np.inf
