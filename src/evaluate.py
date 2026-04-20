import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def dice_score(pred, target, smooth=1e-6):
    pred = (pred > 0.5).float()
    target = target.float()

    intersection = (pred * target).sum(dim=(2, 3))
    denominator = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))

    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return dice.mean().item()

def evaluate_classification(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
    }