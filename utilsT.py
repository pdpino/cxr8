"""Training util functions."""

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import numpy as np
from sklearn.metrics import roc_auc_score
from ignite.metrics import EpochMetric

from dataset import CXRDataset


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

def prepare_data(dataset_dir, dataset_type, chosen_diseases, batch_size, shuffle=False, max_images=None, image_format="RGB"):
    transform_image = get_image_transformation()

    dataset = CXRDataset(dataset_dir,
                         dataset_type=dataset_type,
                         transform=transform_image,
                         diseases=chosen_diseases,
                         max_images=max_images,
                         image_format=image_format)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return dataset, dataloader


def get_image_transformation(image_size=512):
    mean = 0.50576189
    return transforms.Compose([transforms.Resize(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize([mean], [1.])
                              ])