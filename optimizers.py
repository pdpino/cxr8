import torch.optim as optim

class OptimizerNotFound(Exception):
    pass

_OPT_CLASSES = {
    "sgd": optim.SGD,
    "adam": optim.Adam,
}

AVAILABLE_OPTIMIZERS = list(_OPT_CLASSES)

def get_optimizer_class(opt_name):
    if opt_name not in _OPT_CLASSES:
        raise OptimizerNotFound(opt_name)
        
    return _OPT_CLASSES[opt_name]