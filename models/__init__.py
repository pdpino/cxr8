import torch
from torch.nn import DataParallel
import os

import optimizers
from models import v0, v1, v2, v3, v4

_MODELS_DEF = {
    "v0": v0.ResnetBasedModel,
    "v1": v1.ResnetBasedModel,
    "v2": v2.ResnetBasedModel,
    "v3": v3.ResnetBasedModel,
    "v4": v4.DensenetBasedModel,
}

AVAILABLE_MODELS = list(_MODELS_DEF)


def init_empty_model(model_name, chosen_diseases, train_resnet=False, **kwargs):
    if model_name not in _MODELS_DEF:
        raise Exception("Model not found: {}".format(model_name))
    ModelClass = _MODELS_DEF[model_name]

    return ModelClass(n_diseases=len(chosen_diseases), train_resnet=train_resnet, **kwargs)


def get_model_fname(base_dir, run_name, experiment_mode="debug"):
    folder = os.path.join(base_dir, "models")
    if experiment_mode:
        folder = os.path.join(folder, experiment_mode)
    return os.path.join(folder, run_name + ".pth")


def save_model(base_dir, run_name, model_name, experiment_mode, hparam_dict, trainer, model, optimizer):
    model_fname = get_model_fname(base_dir, run_name, experiment_mode=experiment_mode)
    torch.save({
        "model_name": model_name,
        "hparams": hparam_dict,
        "epoch": trainer.state.epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, model_fname)


def load_model(base_dir, run_name, experiment_mode="", device=None, force_multiple_gpu=False):
    model_fname = get_model_fname(base_dir, run_name, experiment_mode=experiment_mode)
    
    checkpoint = torch.load(model_fname)
    hparams = checkpoint["hparams"]
    model_name = checkpoint.get("model_name", "v0")
    chosen_diseases = hparams["diseases"].split(",")
    train_resnet = hparams["train_resnet"]
    multiple_gpu = hparams.get("multiple_gpu", False)
    
    def extract_params(name):
        params = {}
        prefix = name + "_"
        for key, value in hparams.items():
            if key.startswith(prefix):
                key = key[len(prefix):]
                params[key] = value
        return params
    
    opt_params = extract_params("opt")

    # Load model
    model = init_empty_model(model_name, chosen_diseases, train_resnet)
    
    # NOTE: this force param has to be used for cases when the hparam was not saved
    if force_multiple_gpu or multiple_gpu:
        model = DataParallel(model)
    
    if device:
        model = model.to(device)
    
    # Load optimizer
    opt_name = hparams["opt"]
    OptClass = optimizers.get_optimizer_class(opt_name)
    optimizer = OptClass(model.parameters(), **opt_params)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load loss
    loss_name = hparams["loss"]
    loss_params = extract_params("loss")
    
    # TODO: make a class to hold all of these values (and avoid changing a lot of code after any change here)
    return model, model_name, optimizer, opt_name, loss_name, loss_params, chosen_diseases
