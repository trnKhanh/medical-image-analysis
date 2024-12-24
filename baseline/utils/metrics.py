import torch
import torch.nn.functional as F
import torch.nn as nn
import SimpleITK as sitk
import numpy as np

class DSC(nn.Module):
    def __init__(self):
        super().__init__()
        self.smooth = 1e-6

    def forward(self, y_pred, y_truth):
        """
        :param y_pred: (BS,3,336,544)
        :param y_truth: (BS,336,544)
        :return:
        """
        y_pred_f = F.one_hot(y_pred.argmax(dim=1).long(), 3)  # (BS,336,544,3)
        y_pred_f = torch.flatten(y_pred_f, start_dim=0, end_dim=2)   # (-1,3)

        y_truth_f = F.one_hot(y_truth.long(), 3)  # (BS,336,544,3)
        y_truth_f = torch.flatten(y_truth_f, start_dim=0, end_dim=2)  # (-1,3)

        dice1 = (2. * ((y_pred_f[:, 1:2] * y_truth_f[:, 1:2]).sum()) + self.smooth) / (
                y_pred_f[:, 1:2].sum() + y_truth_f[:, 1:2].sum() + self.smooth)
        dice2 = (2. * ((y_pred_f[:, 2:] * y_truth_f[:, 2:]).sum()) + self.smooth) / (
                y_pred_f[:, 2:].sum() + y_truth_f[:, 2:].sum() + self.smooth)

        dice1.requires_grad_(False)
        dice2.requires_grad_(False)
        return dice1,dice2


class HD(nn.Module):
    def __init__(self):
        super(HD,self).__init__()
        pass

    def numpy_to_image(self, image) -> sitk.Image:
        image = sitk.GetImageFromArray(image)
        return image

    def evaluation(self, pred: sitk.Image, label: sitk.Image):
        result = dict()

        # 计算upper指标
        pred_data_upper = sitk.GetArrayFromImage(pred)
        pred_data_upper[pred_data_upper == 2] = 0
        pred_upper = sitk.GetImageFromArray(pred_data_upper)

        label_data_upper = sitk.GetArrayFromImage(label)
        label_data_upper[label_data_upper == 2] = 0
        label_upper = sitk.GetImageFromArray(label_data_upper)

        
        result['hd_upper'] = float(self.cal_hd(pred_upper, label_upper))

        # 计算lower指标
        pred_data_lower = sitk.GetArrayFromImage(pred)
        pred_data_lower[pred_data_lower == 1] = 0
        pred_data_lower[pred_data_lower == 2] = 1
        pred_lower = sitk.GetImageFromArray(pred_data_lower)

        label_data_lower = sitk.GetArrayFromImage(label)
        label_data_lower[label_data_lower == 1] = 0
        label_data_lower[label_data_lower == 2] = 1
        label_lower = sitk.GetImageFromArray(label_data_lower)

   
        
        result['hd_lower'] = float(self.cal_hd(pred_lower, label_lower))

        # 计算总体指标
        pred_data_all = sitk.GetArrayFromImage(pred)
        pred_data_all[pred_data_all == 2] = 1
        pred_all = sitk.GetImageFromArray(pred_data_all)

        label_data_all = sitk.GetArrayFromImage(label)
        label_data_all[label_data_all == 2] = 1
        label_all = sitk.GetImageFromArray(label_data_all)
        
        result['hd_all'] = float(self.cal_hd(pred_all, label_all))
        
        return (result['hd_all'] + result['hd_lower'] + result['hd_upper']) / 3

    def forward(self, pred, label):
        """
        :param pred: (BS,3,336,544)
        :param label: (BS,336,544)
        :return:
        """
        #print(pred.shape)
        #print(label.shape)

        pred = torch.argmax(pred,dim=1)[0].detach().cpu().numpy().astype(np.int64)  # (H,W) value:0,1,2  1-upper 2-lower
        label = label[0].detach().cpu().numpy().astype(np.int64) # (H,W) value:0,1,2  1-upper 2-lower
        pre_image = self.numpy_to_image(pred)
        truth_image = self.numpy_to_image(label)
        result = self.evaluation(pre_image, truth_image)

        return result


    def cal_hd(self, a, b):
        a = sitk.Cast(sitk.RescaleIntensity(a), sitk.sitkUInt8)
        b = sitk.Cast(sitk.RescaleIntensity(b), sitk.sitkUInt8)
        filter1 = sitk.HausdorffDistanceImageFilter()
        filter1.Execute(a, b)
        hd = filter1.GetHausdorffDistance()
        return hd