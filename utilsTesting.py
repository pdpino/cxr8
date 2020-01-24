import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch


def image_to_range(image, min_value=0, max_value=1):
    return np.interp(image, (image.min(), image.max()), (min_value, max_value))


def gen_image_with_bbox(model, dataset, image_name, chosen_diseases, device):
    """TODO.
    
    Notice that chosen_diseases and dataset.classes may be different.
    """
    bboxes = []

    image, labels, _, bboxes_raw, are_valid = dataset.get_by_name(image_name, chosen_diseases)

    # Convert to batch
    images = image.view(1, *image.shape)

    # Image to GPU
    images = images.to(device)

    # Pass thru model
    with torch.no_grad():
        predictions, _, activations = model(images)

    # Copy bbox
    # TODO: this bbox handling could come from dataset itself (simplifies the valid thing)
    for disease_name in chosen_diseases:
        disease_index = dataset.classes.index(disease_name) # Index in dataset
        
        bbox = bboxes_raw[disease_index]
        is_valid = are_valid[disease_index]
        is_valid = bool(is_valid.item())
        if is_valid:
            x, y, w, h = bbox.numpy()
            bboxes.append((disease_name, x, y, w, h))

    # If activations
    if activations is not None:
        activations = activations[0].cpu().numpy()

    return image.numpy(), labels, predictions[0].cpu().numpy(), bboxes, activations

colors = ["red","blue","cyan","green"]

def plot_bboxes(bboxes, scale=2):
    # Add bboxes
    ax = plt.gca()
    for index, bbox in enumerate(bboxes):
        disease, x, y, width, height = bbox

        x /= scale
        y /= scale
        width /= scale
        height /= scale

        color = colors[index]
        rect = patches.Rectangle((x, y), width, height,
                                 linewidth=1, edgecolor=color, facecolor="none", label=disease)
        ax.add_patch(rect)

    if len(bboxes) > 0:
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        
def plot_image_with_bbox(image, image_name, bboxes, scale=2):
    """Plot image with its bboxes.
    
    scale: factor to reduce the bboxes
    """
    # Plot image
    plt.title(image_name)
    norm_image_CHW = image_to_range(image)
    norm_image_HWC = norm_image_CHW.transpose(1, 2, 0)
    plt.imshow(norm_image_HWC)
    
    
    plot_bboxes(bboxes)


    
def plot_activation(activations, prediction, gt, chosen_diseases, disease_name=None):
    if disease_name is None:
        disease_name = chosen_diseases[0]

    disease_index = chosen_diseases.index(disease_name)

    # pred = prediction[disease_index]

    # FIXME: this title is printed outside this function (remove unused params)
    # plt.title("{} ({}, {:.4f})".format(disease_name, gt[disease_index], pred))

    plt.title("Network activation (CAM)")
    plt.imshow(activations[disease_index], cmap="Blues")
    plt.colorbar()
