from datetime import datetime
import time

BASE_DIR = "/mnt/data/chest-x-ray-8" # In server
DATASET_DIR = BASE_DIR + "/dataset"
CMS_DIR = BASE_DIR + "/cms"

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


def write_to_txt(arr, fname, sep='\n'):
    """Writes a list of strings to a file"""
    with open(fname, 'w') as f:
        for line in arr:
            f.write(line + sep)


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
    all_image_names = [0] * n_images

    batch_size = None
    image_index = 0

    for data_batch in dataloader:
        images, labels, image_names, _, _ = data_batch

        images = images.to(device)

        current_batch_size = images.size()[0]

        if batch_size is None:
            batch_size = current_batch_size

        index_from = image_index * batch_size
        index_to = index_from + current_batch_size
        
        if index_from >= n_images:
            break

        with torch.no_grad():
            predictions, _, _ = model(images)

        all_predictions[index_from:index_to] = predictions.cpu()
        all_ground_truths[index_from:index_to] = torch.Tensor(labels.float())
        all_image_names[index_from:index_to] = image_names
        
        image_index += 1


    return all_predictions.numpy(), all_ground_truths.numpy(), all_image_names


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


def plot_cm(cm, classes, title, percentage=False):
    n_classes = len(classes)
    ticks = np.arange(n_classes)

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()

    plt.xticks(ticks, classes)
    plt.yticks(ticks, classes)
    plt.title(title)
    plt.xlabel("Prediction")
    plt.ylabel("True")

    total = cm.sum()
    
    thresh = cm.max() / 2
    for row in range(n_classes):
        for col in range(n_classes):
            value = cm[row, col]
            color = "white" if value > thresh else "black"
            
            value_str = "{:d}".format(int(value))
            if percentage:
                value_str += "\n({:.3f})".format(value/total)
                
            plt.text(col, row, value_str, ha="center", va="center", color=color)


def plot_cms(cms, classes, diseases, n_cols=3):
    n_diseases = len(diseases)
    
    n_rows = math.ceil(n_diseases/n_cols)
    
    for index, disease in enumerate(diseases):
        cm = cms[index]
        plt.subplot(n_rows, n_cols, index+1)
        plot_cm(cm, classes, disease)
        
        
def plot_train_val_cms(train_cms, val_cms, classes, diseases, percentage=False):
    n_diseases = len(diseases)
    
    n_cols = 2
    n_rows = math.ceil(n_diseases*2/n_cols)
    
    plot_index = 1
    for index, disease in enumerate(diseases):
        train_cm = train_cms[index]
        val_cm = val_cms[index]

        plt.subplot(n_rows, n_cols, plot_index)
        plot_cm(train_cm, classes, disease, percentage=percentage)
        plot_index += 1
        
        plt.subplot(n_rows, n_cols, plot_index)
        plot_cm(val_cm, classes, disease, percentage=percentage)
        plot_index += 1
        
    print("Left: training")
    print("Right: validation")
        
## CMS with names (i.e. which images are TP, FP, TN, FN)
def calculate_all_cms_names(all_predictions, all_ground_truths, image_names, chosen_diseases):
    n_diseases = all_predictions.shape[1]

    def get_names(condition):
        condition = condition.flatten()
        return [image_name for image_name, is_good in zip(image_names, condition) if is_good]
    
    thresh = 0.5

    cms = []

    for disease_index, disease_name in enumerate(chosen_diseases):
        preds = all_predictions[:, disease_index]
        gts = all_ground_truths[:, disease_index]

        TP = get_names((preds > thresh)  & (gts > thresh))
        FP = get_names((preds > thresh)  & (gts <= thresh))
        TN = get_names((preds <= thresh) & (gts <= thresh))
        FN = get_names((preds <= thresh) & (gts > thresh))
        cms.append({
            "TP": TP,
            "FP": FP,
            "TN": TN,
            "FN": FN,
        })
        
    return cms


import json

def save_cms_names(cms, base_fname, chosen_diseases):
    for cm, disease_name in zip(cms, chosen_diseases):
        with open(base_fname + "_" + disease_name + ".json", "w") as f:
            json.dump(cm, f)
        

def load_cms_names(fname, disease_name):
    with open(fname + "_" + disease_name + ".json") as f:
        cms_names = json.load(f)
        
    TP = cms_names["TP"]
    FP = cms_names["FP"]
    TN = cms_names["TN"]
    FN = cms_names["FN"]
    
    cm = np.array([
        [len(TN), len(FP)],
        [len(FN), len(TP)],
    ])

    return TP, FP, TN, FN, cm