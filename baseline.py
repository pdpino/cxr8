"""Run dummy baselines."""

import torch
# from torch.nn import DataParallel
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from torchvision import transforms
# from tensorboardX import SummaryWriter
from ignite.engine import Engine # , Events
from ignite.metrics import Accuracy, Precision, Recall, RunningAverage # , ConfusionMatrix, VariableAccumulation #, EpochMetric
from ignite.handlers import Timer #, EarlyStopping
# from ignite.utils import to_onehot

import numpy as np
import argparse
import time
import os
import json
# import warnings

# from dataset import CXRDataset
# import models
# import losses
# import optimizers
import utils
import utilsT
# from test import save_cms_with_names, evaluate_model


ALL_METRICS = ["roc_auc", "prec", "recall", "acc"]


def get_step_fn_label(chosen_label):
    def step_fn(engine, data_batch):
        _, labels, _, _, _ = data_batch

        outputs = torch.Tensor().new_full(labels.size(), chosen_label)

        dummy_batch_loss = 0

        return dummy_batch_loss, outputs, labels

    return step_fn

def get_step_fn_random():
    def step_fn(engine, data_batch):
        _, labels, _, _, _ = data_batch

        outputs = np.random.choice([0, 1], labels.shape, replace=True)
        outputs = torch.Tensor(outputs)

        dummy_batch_loss = 0

        return dummy_batch_loss, outputs, labels
    
    return step_fn


def test_baseline_model(baseline,
                        chosen_label=0,
                        name="",
                        base_dir=utils.BASE_DIR,
                        chosen_diseases=None,
                        dataset_type="test",
                        batch_size=4,
                        max_images=None,
                        experiment_mode="debug",
                       ):
    # Common folders
    dataset_dir = os.path.join(base_dir, "dataset")

    # Dataset handling
    print("Loading {} dataset...".format(dataset_type))
    dataset, dataloader = utilsT.prepare_data(
        dataset_dir, dataset_type, chosen_diseases, batch_size, max_images=max_images)

    # Should be the same than chosen_diseases
    chosen_diseases = list(dataset.classes)
    print("Chosen diseases: ", chosen_diseases)

    # Tensorboard log options
    run_name = utils.get_timestamp()
    if name:
        run_name += "_{}".format(name)

    if len(chosen_diseases) == 1:
        run_name += "_{}".format(chosen_diseases[0])
    elif len(chosen_diseases) == 14:
        run_name += "_all"
        
    if baseline == "label":
        run_name += "_label{}".format(chosen_label)
    elif baseline == "random":
        run_name += "_random"
    else:
        raise Exception("Baseline not found: {}".format(baseline))

    print("Run name: ", run_name)

    # Create tester engine
    if baseline == "label":
        step_fn = get_step_fn_label(chosen_label)
    elif baseline == "random":
        step_fn = get_step_fn_random()
    
    tester = Engine(step_fn)
    
    utilsT.attach_metrics(tester, chosen_diseases, "acc", Accuracy, True)
    utilsT.attach_metrics(tester, chosen_diseases, "prec", Precision, True)
    utilsT.attach_metrics(tester, chosen_diseases, "recall", Recall, True)
    utilsT.attach_metrics(tester, chosen_diseases, "roc_auc", utilsT.RocAucMetric, False)

    # Test
    print("-" * 50)
    print("Test...")
    tester.run(dataloader, 1)

    
    # Copy metrics dict
    metrics = dict()
    original_metrics = tester.state.metrics
    for metric_name in ALL_METRICS:
        for disease_name in chosen_diseases:
            key = metric_name + "_" + disease_name
            if key not in original_metrics:
                print("Metric not found in tester, skipping: ", key)
                continue

            metrics[key] = original_metrics[key]
    
    # Save to file
    folder = os.path.join(base_dir, "results", experiment_mode)
    os.makedirs(folder, exist_ok=True)
    
    fname = os.path.join(folder, run_name + ".json")
    with open(fname, "w+") as f:
        json.dump(metrics, f)   
    print("Saved metrics to: ", fname)
    


def parse_args():
    parser = argparse.ArgumentParser(description="Run a dummy baseline", usage="%(prog)s [options]")
    
    
    parser.add_argument("baseline", type=str, default=None, choices=["random", "label"],
                        help="Choose the baseline")
    parser.add_argument("--chosen-label", type=int, default=0,
                        help="If baseline label is used, always predict with the fixed label provided (0 or 1)")
    parser.add_argument("--name", default="", type=str, help="Additional name to the run")
    parser.add_argument("--base-dir", default=utils.BASE_DIR, type=str, help="Base folder")
    parser.add_argument("--diseases", default=None, nargs="*", type=str, choices=utils.ALL_DISEASES,
                        help="Diseases to train with")
    parser.add_argument("--n-diseases", default=None, type=int, help="If present, select the first n diseases")
    parser.add_argument("-bs", "--batch-size", default=4, type=int, help="Batch size")
    parser.add_argument("--image-format", default="RGB", type=str, help="Image format passed to Pillow")
    parser.add_argument("--max-images", default=None, type=int, help="Max images")
    parser.add_argument("--non-debug", default=False, action="store_true",
                        help="If present, is considered an official model")
    
    args = parser.parse_args()
    
    # If debugging, automatically set other values
    is_debug = not args.non_debug
    if is_debug:
        args.max_images = args.max_images or 100

    # Shorthand for first diseases
    if args.n_diseases and args.n_diseases > 0:
        args.diseases = utils.ALL_DISEASES[:args.n_diseases]
    
    return args
    
if __name__ == "__main__":
    args = parse_args()
    
    experiment_mode = "debug"
    if args.non_debug:
        experiment_mode = ""
        
    start_time = time.time()
    
    test_baseline_model(args.baseline,
                        chosen_label=args.chosen_label,
                        name=args.name,
                        base_dir=args.base_dir,
                        chosen_diseases=args.diseases,
                        batch_size=args.batch_size,
                        max_images=args.max_images,
                        experiment_mode=experiment_mode,
                       )
    
    end_time = time.time()
    
    print("-"*50)
    print("Total time: ", utils.duration_to_str(end_time - start_time))
    print("="*50)