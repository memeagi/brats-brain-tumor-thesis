import torch
import torch.nn as nn

bce_loss = nn.BCELoss()

def dice_loss(pred, target, smooth=1e-6):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=(2, 3))
    denominator = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))

    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return 1 - dice.mean()

def combined_bce_dice_loss(pred, target):
    bce = bce_loss(pred, target)
    dice = dice_loss(pred, target)
    return bce + dice