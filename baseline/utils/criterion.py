from .loss import *


class MyCriterion(nn.Module):
    def __init__(self):
        super(MyCriterion, self).__init__()
        self.DC = DiceLoss()
        self.CE = CELoss()

    def forward(self, pred, label):
        """

        :param pred: (BS,3,336,544)
        :param label: (BS,336,544)
        :return:
        """
        label_onehot = F.one_hot(label.long(), 3) # (BS, 336, 544, 3)
        pred_t = F.one_hot(pred.argmax(dim=1).long(), 3) # (BS, 336, 544, 3)
        dice_loss = self.DC(pred_t, label_onehot)
        ce_loss = self.CE(pred, label_onehot)
        print(f" DC: {dice_loss.detach().cpu().numpy():.5f} || CE: {ce_loss.detach().cpu().numpy():.5f}")
        supervised_loss = 0.5 * ce_loss + 0.5 * dice_loss
        return supervised_loss


class ClsCriterion(nn.Module):
    def __init__(self):
        super(ClsCriterion, self).__init__()

    def forward(self, pred, label):
        ce_loss = F.cross_entropy(pred, label)
        print(f" CE: {ce_loss.detach().cpu().numpy():.4f}")
        return ce_loss