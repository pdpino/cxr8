# import torch
from ignite.engine import Engine, Events
from ignite.metrics import Accuracy, Precision, Recall, RunningAverage, ConfusionMatrix, VariableAccumulation
from ignite.handlers import Timer
from ignite.utils import to_onehot

import numpy as np
import argparse
import time
import os
import warnings
import json

import models
import utils
import utilsT
# from train import get_step_fn, attach_metrics, get_transform_cm, get_count_positives


# TODO: move this to utils?
ALL_METRICS = ["roc_auc", "prec", "recall", "acc"]


def save_cms_with_names(run_name, experiment_mode, model, dataset, dataloader, chosen_diseases):
    n_images = dataset.size()[0]
    n_diseases = len(chosen_diseases)
    device = utilsT.get_torch_device()
    
    print("Predicting all...")
    predictions, gts, image_names = utils.predict_all(
        model, dataloader, device, n_images, n_diseases)

    print("Calculating CM...")
    cms_names = utils.calculate_all_cms_names(predictions, gts, image_names, chosen_diseases)

    fname = os.path.join(utils.CMS_DIR,
                         experiment_mode,
                         run_name + "_" + dataset.dataset_type + "_names",
                        )
    utils.save_cms_names(cms_names, fname, chosen_diseases)


def evaluate_model(run_name, model, optimizer, device, loss_name, loss_params, chosen_diseases, dataloader,
                   experiment_mode="debug", base_dir=utils.BASE_DIR):
    # Create tester engine
    tester = Engine(utilsT.get_step_fn(model, optimizer, device, loss_name, loss_params, training=False))

    loss_metric = RunningAverage(output_transform=lambda x: x[0], alpha=1)
    loss_metric.attach(tester, loss_name)
    
    utilsT.attach_metrics(tester, chosen_diseases, "prec", Precision, True)
    utilsT.attach_metrics(tester, chosen_diseases, "recall", Recall, True)
    utilsT.attach_metrics(tester, chosen_diseases, "acc", Accuracy, True)
    utilsT.attach_metrics(tester, chosen_diseases, "roc_auc", utilsT.RocAucMetric, False)
    utilsT.attach_metrics(tester, chosen_diseases, "cm", ConfusionMatrix,
                          get_transform_fn=utilsT.get_transform_cm, metric_args=(2,))

    timer = Timer(average=True)
    timer.attach(tester, start=Events.EPOCH_STARTED, step=Events.EPOCH_COMPLETED)

    # Save metrics
    log_metrics = list(ALL_METRICS)

    # Run test
    print("Testing...")
    tester.run(dataloader, 1)
    

    # Capture time
    secs_per_epoch = timer.value()
    duration_per_epoch = utils.duration_to_str(int(secs_per_epoch))
    print("Time elapsed in epoch: ", duration_per_epoch)

    # Copy metrics dict
    metrics = dict()
    original_metrics = tester.state.metrics
    for metric_name in log_metrics:
        for disease_name in chosen_diseases:
            key = metric_name + "_" + disease_name
            if key not in original_metrics:
                print("Metric not found in tester, skipping: ", key)
                continue

            metrics[key] = original_metrics[key]
            
    # Copy CMs
    for disesase_name in chosen_diseases:
        key = "cm_" + disease_name
        if key not in original_metrics:
            print("CM not found in tester, skipping: ", key)
            continue
        
        cm = original_metrics[key]
        metrics[key] = cm.numpy().tolist()
    
    # Save to file
    folder = os.path.join(base_dir, "results", experiment_mode)
    os.makedirs(folder, exist_ok=True)
    
    fname = os.path.join(folder, run_name + ".json")
    with open(fname, "w+") as f:
        json.dump(metrics, f)   
    print("Saved metrics to: ", fname)
    
    return metrics

def main(name,
         experiment_mode="debug",
         base_dir=utils.BASE_DIR,
         dataset_type="test",
         max_images=None,
         batch_size=4,
         image_format="RGB",
        ):
    
    # Choose GPU
    device = utilsT.get_torch_device()
    print("Using device: ", device)
    
    # Common folders
    dataset_dir = os.path.join(base_dir, "dataset")

    # Load model and optimizer
    model, model_name, optimizer, opt_name, loss_name, loss_params, chosen_diseases = models.load_model(
        base_dir, name, experiment_mode=experiment_mode, device=device)
    model.eval()
    
    # Dataset handling
    print("Loading {} dataset...".format(dataset_type))
    dataset, dataloader = utilsT.prepare_data(
        dataset_dir, dataset_type, chosen_diseases, batch_size, shuffle=False,
        max_images=max_images, image_format=image_format)
    n_samples, _ = dataset.size()

    save_cms_with_names(name, experiment_mode, model, dataset, dataloader, chosen_diseases)
    metrics = evaluate_model(name, model, optimizer, device, loss_name, loss_params, chosen_diseases, dataloader,
                             experiment_mode=experiment_mode, base_dir=base_dir)
    
    return metrics

    
def parse_args():
    parser = argparse.ArgumentParser(description="Train a model", usage="%(prog)s [options]")
    
    parser.add_argument("--run-name", default="", type=str, help="Name of the model to run")
    parser.add_argument("--base-dir", default=utils.BASE_DIR, type=str, help="Base folder")
    parser.add_argument("--dataset-type", default="test", type=str, choices=["train", "test", "val"],
                        help="Dataset type to use for testing")
    parser.add_argument("-bs", "--batch-size", default=4, type=int, help="Batch size")
    parser.add_argument("--image-format", default="RGB", type=str, help="Image format passed to Pillow")
    parser.add_argument("--max-images", default=None, type=int, help="Max images tested")
    parser.add_argument("--non-debug", default=False, action="store_true",
                        help="If present, it won't go the debug folder")
    
    args = parser.parse_args()
    
    # If debugging, automatically set other values
    is_debug = not args.non_debug
    if is_debug:
        args.max_images = args.max_images or 100
    
    return args
    
if __name__ == "__main__":
    args = parse_args()

    experiment_mode = "debug"
    if args.non_debug:
        experiment_mode = ""

    start_time = time.time()

    metrics = main(args.run_name,
                   experiment_mode=experiment_mode,
                   base_dir=args.base_dir,
                   dataset_type=args.dataset_type,
                   max_images=args.max_images,
                   image_format=args.image_format,
                   batch_size=args.batch_size,
                  )
    end_time = time.time()
    
    print("-"*50)
    print("Total testing time: ", utils.duration_to_str(end_time - start_time))
    print("="*50)