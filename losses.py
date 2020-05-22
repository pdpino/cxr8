import torch
import numpy as np


class LossNotFound(Exception):
    pass


def bce(output, target, epsilon=1e-5):
    """Computes binary cross entropy loss.
    
    If a multi-label array is given, the BCE is summed across labels.
    """
    output = output.clamp(min=epsilon, max=1-epsilon)
    target = target.float()

    loss = -target * torch.log(output) - (1 - target) * torch.log(1 - output)

    return torch.sum(loss)


def weighted_bce(output, target, epsilon=1e-5):
    """Computes weighted binary cross entropy loss.
    
    If a multi-label array is given, the BCE is summed across labels.
    Note that the BP and BN weights are calculated by batch, not in the whole dataset.
    """
    output = output.clamp(min=epsilon, max=1-epsilon)
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


def weighted_bce_by_disease(output, target, epsilon=1e-5):
    output = output.clamp(min=epsilon, max=1-epsilon)
    target = target.float()

    batch_size, n_diseases = target.size()
    
    total = torch.Tensor().new_full((n_diseases,), batch_size).type(torch.float)
    positive = torch.sum(target > 0, dim=0).type(torch.float)
    negative = total - positive

    # If a value is zero, is set to batch_size (so the division results in 1 for that disease)
    positive = positive + ((positive == 0)*batch_size).type(positive.dtype)
    negative = negative + ((negative == 0)*batch_size).type(negative.dtype)
    
    BP = total / positive
    BN = total / negative
    
    loss = -BP * target * torch.log(output) - BN * (1 - target) * torch.log(1 - output)

    return torch.sum(loss)


def focal_loss(output, target, alpha=0.75, gamma=2, epsilon=1e-5):
    """Computes focal loss.
    
    If a multi-label array is given, the loss is summed across labels.
    Based on this post: https://leimao.github.io/blog/Focal-Loss-Explained/
    """
    output = output.clamp(min=epsilon, max=1-epsilon)
    target = target.float()

    # Calculate p_t
    # Note that (for each label) only one term will survive, either output or (1-output)
    pt = target * output + (1 - target) * (1 - output)
    
    # Calculate log(p_t)
    # It could also be calculated as torch.log(pt)
    log_pt = target * torch.log(output) + (1 - target) * torch.log(1 - output)

    # Calculate other terms
    alpha_t = target * alpha + (1-target)*(1 - alpha) # Only one term survives
    term_gamma = (1 - pt)**gamma
    
    loss = - alpha_t * term_gamma * log_pt

    return torch.sum(loss)


_LOSS_FNS = {
    "bce": bce,
    "wbce": weighted_bce,
    "wbce_loss": weighted_bce,
    "wbce_by_disease": weighted_bce_by_disease,
    "focal": focal_loss,
}

AVAILABLE_LOSSES = list(_LOSS_FNS)

def get_loss_function(loss_name, **loss_params):
    if loss_name not in _LOSS_FNS:
        raise LossNotFound(loss_name)
        
    loss_fn = _LOSS_FNS[loss_name]
        
    def loss_wrapper(output, target):
        return loss_fn(output, target, **loss_params)
    
    return loss_wrapper