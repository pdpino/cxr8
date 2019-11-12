"""Training util functions."""

import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from ignite.metrics import EpochMetric

def get_torch_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

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