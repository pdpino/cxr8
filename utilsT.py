"""Training util functions."""

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import numpy as np
from sklearn.metrics import roc_auc_score
from ignite.metrics import EpochMetric
from ignite.utils import to_onehot

from dataset import CXRDataset, CXRUnbalancedSampler
import losses


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

def prepare_data(dataset_dir, dataset_type, chosen_diseases, batch_size, oversample=False,
                 shuffle=False, max_images=None, image_format="RGB"):
    transform_image = get_image_transformation()

    dataset = CXRDataset(dataset_dir,
                         dataset_type=dataset_type,
                         transform=transform_image,
                         diseases=chosen_diseases,
                         max_images=max_images,
                         image_format=image_format)

    if oversample:
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=CXRUnbalancedSampler(dataset))
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return dataset, dataloader


def get_image_transformation(image_size=512):
    mean = 0.50576189
    return transforms.Compose([transforms.Resize(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize([mean], [1.])
                              ])

def get_step_fn(model, optimizer, device, loss_name, loss_params={}, training=True):
    """Creates a step function for an Engine."""
    loss_fn = losses.get_loss_function(loss_name, **loss_params)
    
    def step_fn(engine, data_batch):
        # Input and sizes
        images, labels, names, _, _ = data_batch
        n_samples, n_labels = labels.size()
        
        # Move tensors to GPU
        images = images.to(device)
        labels = labels.to(device)

        # Enable training
        model.train(training)
        torch.set_grad_enabled(training) # enable recording gradients

        # zero the parameter gradients
        optimizer.zero_grad()

        # Forward, receive outputs from the model and segments (bboxes)
        outputs, embedding_unused, segments_unused = model(images)

#         if bool(torch.isnan(outputs).any().item()):
#             warnings.warn("Nans found in prediction")
#             outputs[outputs != outputs] = 0 # Set NaNs to 0
        
        # Compute classification loss
        loss = loss_fn(outputs, labels)

#         if bool(torch.isnan(loss).any().item()):
#             warnings.warn("Nans found in loss: {}".format(loss))
#             loss[loss != loss] = 0
        
        batch_loss = loss.item()

        if training:
            loss.backward()
            optimizer.step()

        return batch_loss, outputs, labels

    return step_fn


def get_transform_one_label(label_index, use_round=True):
    """Creates a transform function to extract one label from a multi-label output."""
    def transform_fn(output):
        _, y_pred, y_true = output
        
        y_pred = y_pred[:, label_index]
        y_true = y_true[:, label_index]
        
        if use_round:
            y_pred = torch.round(y_pred)

        return y_pred, y_true
    return transform_fn


def get_transform_cm(label_index, num_classes=2):
    """Creates a transform function to prepare the input for the ConfusionMatrix metric."""
    def transform_fn(output):
        _, y_pred, y_true = output
        
        # print("ORIGINAL: ", torch.round(y_pred[:, label_index]).long())
        y_pred = to_onehot(torch.round(y_pred[:, label_index]).long(), num_classes)
        y_true = y_true[:, label_index]
        
        # print("TO: ", y_pred)

        return y_pred, y_true
    
    return transform_fn


def get_count_positives(label_index):
    def count_positives_fn(result):
        """Count positive examples in a batch (for a given disease index)."""
        _, _, labels = result
        return torch.sum(labels[:, label_index]).item()

    return count_positives_fn


def attach_metrics(engine, chosen_diseases, metric_name, MetricClass,
                   use_round=True,
                   get_transform_fn=None,
                   metric_args=()):
    """Attaches one metric per label to an engine."""
    for index, disease in enumerate(chosen_diseases):
        if get_transform_fn:
            transform_disease = get_transform_fn(index)
        else:
            transform_disease = get_transform_one_label(index, use_round=use_round)

        metric = MetricClass(*metric_args, output_transform=transform_disease)
        metric.attach(engine, "{}_{}".format(metric_name, disease))
