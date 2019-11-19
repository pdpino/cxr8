import torch

import os

import optimizers
from models import v0, v1, v2
# from .v0 import ResnetBasedModel



_MODELS_DEF = {
    "v0": v0.ResnetBasedModel,
    "v1": v1.ResnetBasedModel,
    "v2": v2.ResnetBasedModel,
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


def load_model(base_dir, run_name, experiment_mode="", device=None):
    model_fname = get_model_fname(base_dir, run_name, experiment_mode=experiment_mode)
    
    checkpoint = torch.load(model_fname)
    hparams = checkpoint["hparams"]
    model_name = hparams.get("model_name", "v0")
    chosen_diseases = hparams["diseases"].split(",")
    train_resnet = hparams["train_resnet"]
    
    def get_opt_params():
        params = {}
        for key, value in hparams.items():
            if key.startswith("opt_"):
                key = key[4:]
                params[key] = value
        return params
    
    opt_params = get_opt_params()

    # Load model
    model = init_empty_model(model_name, chosen_diseases, train_resnet)
    if device:
        model = model.to(device)
    
    # Load optimizer
    opt = hparams["opt"]
    OptClass = optimizers.get_optimizer_class(opt)
    optimizer = OptClass(model.parameters(), **opt_params)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return model, model_name, optimizer, opt, chosen_diseases
