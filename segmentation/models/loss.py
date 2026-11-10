import torch
import torch.nn as nn

class BCEAndDiceLoss(nn.Module):

    def __init__(self, weight=1, smooth=1e-06):
        super(BCEAndDiceLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.weight = weight
        self.smooth = smooth

    def forward(self, outputs, targets):
        targets = targets.float()
        bce = self.bce_loss(outputs, targets)
        outputs = torch.sigmoid(outputs)
        intersection = torch.sum(targets * outputs)
        dice = 1 - (2.0 * intersection + self.smooth) / (torch.sum(targets) + torch.sum(outputs) + self.smooth)
        total_loss = bce + self.weight * dice
        return total_loss
