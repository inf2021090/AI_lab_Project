import torch
import numpy as np

def accuracy(y_true, y_pred):
    """
    Calculate accuracy
    """
    correct = (y_true == y_pred).sum().item()
    total = y_true.size(0) * y_true.size(1) * y_true.size(2)
    acc = correct / total
    return acc

def precision(y_true, y_pred):
    """
    Calculate precision
    """
    tp = ((y_true == 1) & (y_pred == 1)).sum().item()
    fp = ((y_true == 0) & (y_pred == 1)).sum().item()
    prec = tp / (tp + fp)
    return prec

def recall(y_true, y_pred):
    """
    Calculate recall
    """
    tp = ((y_true == 1) & (y_pred == 1)).sum().item()
    fn = ((y_true == 1) & (y_pred == 0)).sum().item()
    rec = tp / (tp + fn)
    return rec

def f1_score(y_true, y_pred):
    """
    Calculate F1 score
    """
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    f1 = 2 * (prec * rec) / (prec + rec)
    return f1

def jaccard_coef(y_true, y_pred):
    """
    Calculate Jaccard coefficient
    """
    # flatten the tensors
    y_true_flatten = y_true.view(-1)
    y_pred_flatten = y_pred.view(-1)

    # calculate I and U
    intersection = torch.sum(y_true_flatten * y_pred_flatten)
    union = torch.sum(y_true_flatten) + torch.sum(y_pred_flatten) - intersection

    # calculate Jaccard coefficient
    jaccard = (intersection + 1.0) / (union + 1.0)

    return jaccard


