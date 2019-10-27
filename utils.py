from datetime import datetime
import time

ALL_DISEASES = [
    'Atelectasis',
    'Cardiomegaly',
    'Effusion',
    'Infiltration',
    'Mass',
    'Nodule',
    'Pneumonia',
    'Pneumothorax',
    'Consolidation',
    'Edema',
    'Emphysema',
    'Fibrosis',
    'Pleural_Thickening',
    'Hernia',
]

DISEASE_INDEX = { name: index for index, name in enumerate(ALL_DISEASES) }

def get_timestamp():
    return datetime.fromtimestamp(time.time()).strftime('%m%d_%H%M%S')
#     return datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')


def duration_to_str(all_seconds):
    seconds = all_seconds % 60
    minutes = all_seconds // 60
    hours = minutes // 60
    
    minutes = minutes % 60

    return "{}h {}m {}s".format(hours, minutes, int(seconds))


# TODO: separate this file?

import math
import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def predict_all(model, dataloader, device, n_images, n_diseases):
    """Generate all predictions for a dataloader.
    Returns all predictions and ground truths.
    """
    all_predictions = torch.zeros(n_images, n_diseases)
    all_ground_truths = torch.zeros(n_images, n_diseases)

    batch_size = None
    image_index = 0

    for data_batch in dataloader:
        images, labels, _, _, _ = data_batch

        images = images.to(device)

        current_batch_size = images.size()[0]

        if batch_size is None:
            batch_size = current_batch_size

        with torch.no_grad():
            predictions, _, _ = model(images)

        index_from = image_index * batch_size
        index_to = index_from + current_batch_size

        all_predictions[index_from:index_to] = predictions.cpu()
        all_ground_truths[index_from:index_to] = torch.Tensor(labels.float())
        
        image_index += 1


    return all_predictions.numpy(), all_ground_truths.numpy()


### Confusion matrices
def calculate_cm(all_predictions, all_ground_truths, disease_index):
    preds = all_predictions[:, disease_index]
    gts = all_ground_truths[:, disease_index]

    return confusion_matrix(gts, preds > 0.5, labels=[1, 0])


def calculate_all_cms(all_predictions, all_ground_truths):
    n_diseases = all_predictions.shape[1]

    all_cms = []
    for i_disease in range(n_diseases):
        cm = calculate_cm(all_predictions, all_ground_truths, i_disease)
        all_cms.append(cm)
        
    return np.array(all_cms)


def plot_cm(cm, classes, title):
    n_classes = len(classes)
    ticks = np.arange(n_classes)

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()

    plt.xticks(ticks, classes)
    plt.yticks(ticks, classes)
    plt.title(title)
    plt.xlabel("Prediction")
    plt.ylabel("True")

    thresh = cm.max() / 2
    for row in range(n_classes):
        for col in range(n_classes):
            value = cm[row, col]
            color = "white" if value > thresh else "black"
            plt.text(col, row, "{:d}".format(value), ha="center", va="center", color=color)


def plot_cms(cms, classes, diseases):
    n_diseases = len(diseases)
    
    n_cols = 3
    n_rows = math.ceil(n_diseases/n_cols)
    
    for index, disease in enumerate(diseases):
        cm = cms[index]
        plt.subplot(n_rows, n_cols, index+1)
        plot_cm(cm, classes, disease)