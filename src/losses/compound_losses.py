from typing import Callable

import torch
from .dice_loss import MemoryEfficientSoftDiceLoss, DiceLoss
from .ce_loss import RobustCrossEntropyLoss, TopKLoss
from torch import nn


def softmax_helper_dim0(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, 0)


def softmax_helper_dim1(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, 1)


class DiceAndCELoss(nn.Module):
    def __init__(
        self,
        dice_loss: Callable = DiceLoss,
        dice_kwargs: dict = {},
        ce_loss: Callable = RobustCrossEntropyLoss,
        ce_kwargs: dict = {},
        default_dice_weight: float = 1.0,
        default_ce_weight: float = 1.0,
    ):
        super().__init__()
        self.dice_loss = dice_loss(**dice_kwargs)
        self.ce_loss = ce_loss(**ce_kwargs)
        self.default_dice_weight = default_dice_weight
        self.default_ce_weight = default_ce_weight

    def forward(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        dice_weight: float | None = None,
        ce_weight: float | None = None,
    ):
        if not dice_weight:
            dice_weight = self.default_dice_weight

        if not ce_weight:
            ce_weight = self.default_ce_weight

        loss_ce = self.ce_loss(outputs, targets)
        loss_dice = self.dice_loss(outputs, targets)
        loss = ce_weight * loss_ce + dice_weight * loss_dice
        return loss

    def get_dice_loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
    ):
        loss_dice = self.dice_loss(outputs, targets)
        return loss_dice

    def get_ce_loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
    ):
        loss_ce = self.ce_loss(outputs, targets)
        return loss_ce


class DualBranchDiceAndCELoss(nn.Module):
    def __init__(
        self,
        dice_loss: Callable = DiceLoss,
        dice_kwargs: dict = {},
        ce_loss: Callable = RobustCrossEntropyLoss,
        ce_kwargs: dict = {},
        default_dice_weight: float = 0.5,
    ):
        super().__init__()
        self.dice_loss = dice_loss(**dice_kwargs)
        self.ce_loss = ce_loss(**ce_kwargs)
        self.default_dice_weight = default_dice_weight

    def forward(
        self,
        outputs,
        low_res_label_batch,
        dice_weight: float | None = None,
    ):
        if not dice_weight:
            dice_weight = self.default_dice_weight
        # for the first branch
        low_res_logits1 = outputs["low_res_logits1"]
        loss_ce1 = self.ce_loss(low_res_logits1, low_res_label_batch[:].long())
        loss_dice1 = self.dice_loss(
            low_res_logits1, low_res_label_batch, softmax=True
        )
        loss1 = (1 - dice_weight) * loss_ce1 + dice_weight * loss_dice1

        # for the second branch
        low_res_logits2 = outputs["low_res_logits2"]
        loss_ce2 = self.ce_loss(low_res_logits2, low_res_label_batch[:].long())
        loss_dice2 = self.dice_loss(
            low_res_logits2, low_res_label_batch, softmax=True
        )
        loss2 = (1 - dice_weight) * loss_ce2 + dice_weight * loss_dice2

        loss = loss1 + loss2
        return loss, loss1, loss_ce1, loss_dice1, loss2, loss_ce2, loss_dice2


class DC_and_CE_loss(nn.Module):
    def __init__(
        self,
        soft_dice_kwargs,
        ce_kwargs,
        weight_ce=1,
        weight_dice=1,
        ignore_label=None,
        dice_class=MemoryEfficientSoftDiceLoss,
    ):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs["ignore_index"] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(
            apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs
        )

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, (
                "ignore label is not implemented for one hot encoded target variables "
                "(DC_and_CE_loss)"
            )
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = (
            self.dc(net_output, target_dice, loss_mask=mask)
            if self.weight_dice != 0
            else 0
        )
        ce_loss = (
            self.ce(net_output, target[:, 0])
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0)
            else 0
        )

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class DC_and_BCE_loss(nn.Module):
    def __init__(
        self,
        bce_kwargs,
        soft_dice_kwargs,
        weight_ce=1,
        weight_dice=1,
        use_ignore_label: bool = False,
        dice_class=MemoryEfficientSoftDiceLoss,
    ):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!

        target mut be one hot encoded
        IMPORTANT: We assume use_ignore_label is located in target[:, -1]!!!

        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(DC_and_BCE_loss, self).__init__()
        if use_ignore_label:
            bce_kwargs["reduction"] = "none"

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.use_ignore_label = use_ignore_label

        self.ce = nn.BCEWithLogitsLoss(**bce_kwargs)
        self.dc = dice_class(apply_nonlin=torch.sigmoid, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        if self.use_ignore_label:
            # target is one hot encoded here. invert it so that it is True wherever we can compute the loss
            if target.dtype == torch.bool:
                mask = ~target[:, -1:]
            else:
                mask = (1 - target[:, -1:]).bool()
            # remove ignore channel now that we have the mask
            # why did we use clone in the past? Should have documented that...
            # target_regions = torch.clone(target[:, :-1])
            target_regions = target[:, :-1]
        else:
            target_regions = target
            mask = None

        dc_loss = self.dc(net_output, target_regions, loss_mask=mask)
        target_regions = target_regions.float()
        if mask is not None:
            ce_loss = (
                self.ce(net_output, target_regions) * mask
            ).sum() / torch.clip(mask.sum(), min=1e-8)
        else:
            ce_loss = self.ce(net_output, target_regions)
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class DC_and_topk_loss(nn.Module):
    def __init__(
        self,
        soft_dice_kwargs,
        ce_kwargs,
        weight_ce=1,
        weight_dice=1,
        ignore_label=None,
    ):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super().__init__()
        if ignore_label is not None:
            ce_kwargs["ignore_index"] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = TopKLoss(**ce_kwargs)
        self.dc = SoftDiceLoss(
            apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs
        )

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, (
                "ignore label is not implemented for one hot encoded target variables "
                "(DC_and_CE_loss)"
            )
            mask = (target != self.ignore_label).bool()
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = (
            self.dc(net_output, target_dice, loss_mask=mask)
            if self.weight_dice != 0
            else 0
        )
        ce_loss = (
            self.ce(net_output, target)
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0)
            else 0
        )

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result
