"""Training util functions."""

import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from ignite.metrics import EpochMetric

class RocAucWarning(Warning):
    pass

def roc_auc_compute_fn(y_preds, y_true):
    y_true = y_true.numpy()
    if len(np.unique(y_true)) != 2:
        # warnings.warn("ROC AUC = 0", RocAucWarning)
        return 0

    y_pred = y_preds.numpy()

    return roc_auc_score(y_true, y_pred)

def RocAucMetric(**kwargs):
    return EpochMetric(roc_auc_compute_fn, **kwargs)


def weighted_bce(output, target):
    """Computes weighted binary cross entropy loss.
    
    If a multi-label array is given, the BCE is summed across labels."""
    output = output.clamp(min=1e-5, max=1-1e-5)
    target = target.float()

    # Calculate weights
    BP = 1
    BN = 1

    total = np.prod(target.size())
    positive = int((target > 0).sum())
    negative = total - positive

    if positive != 0 and negative != 0:
        BP = total / positive
        BN = total / negative

    loss = -BP * target * torch.log(output) - BN * (1 - target) * torch.log(1 - output)

    return torch.sum(loss)