import torch
# import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter
from ignite.engine import Engine, Events
from ignite.metrics import Accuracy, Precision, Recall, RunningAverage, ConfusionMatrix, VariableAccumulation #, EpochMetric
from ignite.handlers import Timer
from ignite.utils import to_onehot

import numpy as np
import argparse
import time
import os
import warnings

from dataset import CXRDataset
from model import ResnetBasedModel
import utils
import losses
import optimizers
import utilsT


ALL_METRICS = ["roc_auc", "prec", "recall", "acc"]


class ModelRun:
    def __init__(self, model, run_name, chosen_diseases):
        self.model = model
        self.run_name = run_name
        self.chosen_diseases = chosen_diseases
        
        self.writer = None
        self.trainer = None
        self.validator = None
        self.train_dataset = None
        self.train_dataloader = None
        self.val_dataset = None
        self.val_dataloader = None
    
    def save_debug_data(self, writer, trainer, validator, train_dataset, train_dataloader, val_dataset, val_dataloader):
        self.writer = writer
        self.trainer = trainer
        self.validator = validator
        self.train_dataset = train_dataset
        self.train_dataloader = train_dataloader
        self.val_dataset = val_dataset
        self.val_dataloader = val_dataloader



def get_model_fname(base_dir, run_name, experiment_mode="debug"):
    folder = os.path.join(base_dir, "models")
    if experiment_mode:
        folder = os.path.join(folder, experiment_mode)
    return os.path.join(folder, run_name + ".pth")


def get_log_dir(base_dir, run_name, experiment_mode="debug"):
    folder = os.path.join(base_dir, "runs")
    if experiment_mode:
        folder = os.path.join(folder, experiment_mode)
    return os.path.join(folder, run_name)


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


def init_empty_model(chosen_diseases, train_resnet=False):
    return ResnetBasedModel(train_resnet=train_resnet, n_diseases=len(chosen_diseases))


def save_model(base_dir, run_name, experiment_mode, hparam_dict, trainer, model, optimizer):
    model_fname = get_model_fname(base_dir, run_name, experiment_mode=experiment_mode)
    torch.save({
        "hparams": hparam_dict,
        "epoch": trainer.state.epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, model_fname)


def load_model(base_dir, run_name, experiment_mode="", device=None):
    model_fname = get_model_fname(base_dir, run_name, experiment_mode=experiment_mode)
    
    checkpoint = torch.load(model_fname)
    hparams = checkpoint["hparams"]
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
    model = init_empty_model(chosen_diseases, train_resnet)
    if device:
        model = model.to(device)
    
    # Load optimizer
    opt = hparams["opt"]
    OptClass = optimizers.get_optimizer_class(opt)
    optimizer = OptClass(model.parameters(), **opt_params)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return model, optimizer, chosen_diseases


def prepare_data(dataset_dir, dataset_type, chosen_diseases, batch_size, shuffle=False, max_images=None, image_format="RGB"):
    transform_image = get_image_transformation()

    dataset = CXRDataset(dataset_dir,
                         dataset_type=dataset_type,
                         transform=transform_image,
                         diseases=chosen_diseases,
                         max_images=max_images,
                         image_format=image_format)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return dataset, dataloader


def get_image_transformation(image_size=512):
    mean = 0.50576189
    return transforms.Compose([transforms.Resize(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize([mean], [1.])
                              ])


def tb_write_graph(writer, model, dataloader, device):
    # FIXME: Showing this error: https://github.com/lanpa/tensorboardX/issues/483
    images = next(iter(dataloader))[0]
    images = images.to(device)
    writer.add_graph(model, images)


def gen_embeddings(model, dataset, device, image_list=None, image_size=512, n_features=2048):
    if image_list is None:
        image_list = list(dataset.label_index["FileName"])
    
    select_n_images = len(image_list)

    all_embeddings = torch.Tensor(select_n_images, n_features).to(device)
    all_images = torch.Tensor(select_n_images, 3, image_size, image_size).to(device)
    all_predictions = [""] * select_n_images
    all_ground_truths = [""] * select_n_images

    # HACK: this overrides original transform fn! use only at the end of the dataset life
    if image_size > 0:
        dataset.transform = get_image_transformation(image_size)
    
    for index, image_name in enumerate(image_list):
        image, label, _, _, _ = dataset.get_by_name(image_name)

        # Convert to batch
        images = image.view(1, *image.shape)

        # Image to GPU
        images = images.to(device)

        # Pass thru model
        with torch.no_grad():
            predictions, embeddings, _ = model(images)

        # Copy metadata
        all_predictions[index] = predictions.cpu().reshape(-1).numpy()
        all_ground_truths[index] = label

        # Copy embeddings
        all_embeddings[index] = embeddings

        # Copy images
        if image_size > 0:
            all_images[index] = images

        # Has copied index images so far
        if index + 1 >= select_n_images:
            break
            
    
    all_predictions = np.array(all_predictions)
        
    all_ground_truths = np.array(all_ground_truths)
    
    return all_images, all_embeddings, all_predictions, all_ground_truths


def tb_write_embeddings(writer, chosen_diseases,
                        all_images, all_embeddings, all_predictions, all_ground_truths,
                        use_images=True, global_step=None, tag=""):
    label_img = all_images if use_images else None

    metadata_header = []
    for disease in chosen_diseases:
        metadata_header.append("pred_{}".format(disease))
        metadata_header.append("round_{}".format(disease))
        metadata_header.append("gt_{}".format(disease))
        
    metadata = []
    for preds, gts in zip(all_predictions, all_ground_truths):
        mixed = []
        for pred, gt in zip(preds, gts):
            mixed.append("{:.2f}".format(pred))
            mixed.append(round(pred))
            mixed.append(gt)
        metadata.append(mixed)

    writer.add_embedding(all_embeddings,
                         metadata=metadata,
                         label_img=label_img,
                         global_step=global_step,
                         tag=tag,
                         metadata_header=metadata_header,
                        )

def tb_write_cms(writer, dataset_type, diseases, cms):
    for cm, disease in zip(cms, diseases):
        writer.add_text("cm_" + disease + "/" + dataset_type, str(cm))


def tb_write_histogram(writer, model, epoch, wall_time):
#     accept_all = lambda x: True
#     ignore_fn = getattr(model, "ignore_param", accept_all)

    for name, params in model.named_parameters():
#         if ignore_fn(name):
#             continue

        writer.add_histogram(name, params.cpu().detach().numpy(), global_step=epoch, walltime=wall_time)


def train_model(name="",
                resume="",
                base_dir=".",
                chosen_diseases=None,
                n_epochs=10,
                batch_size=4,
                shuffle=False,
                opt="sgd",
                opt_params={},
                loss_name="wbce",
                loss_params={},
                train_resnet=False,
                log_metrics=None, flush_secs=120,
                train_max_images=None, val_max_images=None,
                experiment_mode="debug",
                save=True,
                save_cms=True, # Note that in this case, save_cms (to disk) includes write_cms (to TB)
                write_graph=False,
                write_emb=False,
                write_emb_img=False,
                image_format="RGB",
               ):
    
    # Choose GPU
    device = utilsT.get_torch_device()
    print("Using device: ", device)
    
    # Common folders
    dataset_dir = os.path.join(base_dir, "dataset")

    # Dataset handling
    print("Loading train dataset...")
    train_dataset, train_dataloader = prepare_data(dataset_dir,
                                                   "train",
                                                   chosen_diseases,
                                                   batch_size,
                                                   shuffle=shuffle,
                                                   max_images=train_max_images,
                                                   image_format=image_format,
                                                  )
    train_samples, _ = train_dataset.size()

    print("Loading val dataset...")
    val_dataset, val_dataloader = prepare_data(dataset_dir,
                                               "val",
                                               chosen_diseases,
                                               batch_size,
                                               max_images=val_max_images,
                                               image_format=image_format,
                                              )
    val_samples, _ = val_dataset.size()
    
    # Should be the same than chosen_diseases
    chosen_diseases = list(train_dataset.classes)
    print("Chosen diseases: ", chosen_diseases)

    
    if resume:
        # Load model and optimizer
        model, optimizer, chosen_diseases = load_model(base_dir, resume, experiment_mode="", device=device)
        model.train(True)
    else:
        # Create model
        model = init_empty_model(chosen_diseases, train_resnet=train_resnet).to(device)

        # Create optimizer
        OptClass = optimizers.get_optimizer_class(opt)
        optimizer = OptClass(model.parameters(), **opt_params)
        # print("OPT: ", opt_params)
    
    # Tensorboard log options
    run_name = utils.get_timestamp()
    if name:
        run_name += "_{}".format(name)

    if len(chosen_diseases) == 1:
        run_name += "_{}".format(chosen_diseases[0])
    elif len(chosen_diseases) == 14:
        run_name += "_all"

    log_dir = get_log_dir(base_dir, run_name, experiment_mode=experiment_mode)

    print("Run name: ", run_name)
    print("Saved TB in: ", log_dir)

    writer = SummaryWriter(log_dir=log_dir, flush_secs=flush_secs)
    

    # Create validator engine
    validator = Engine(get_step_fn(model, optimizer, device, loss_name, loss_params, False))

    val_loss = RunningAverage(output_transform=lambda x: x[0], alpha=1)
    val_loss.attach(validator, loss_name)
    
    attach_metrics(validator, chosen_diseases, "prec", Precision, True)
    attach_metrics(validator, chosen_diseases, "recall", Recall, True)
    attach_metrics(validator, chosen_diseases, "acc", Accuracy, True)
    attach_metrics(validator, chosen_diseases, "roc_auc", utilsT.RocAucMetric, False)
    attach_metrics(validator, chosen_diseases, "cm", ConfusionMatrix, get_transform_fn=get_transform_cm, metric_args=(2,))
    attach_metrics(validator, chosen_diseases, "positives", RunningAverage, get_transform_fn=get_count_positives)

    
    # Create trainer engine
    trainer = Engine(get_step_fn(model, optimizer, device, loss_name, loss_params, True))
    
    train_loss = RunningAverage(output_transform=lambda x: x[0], alpha=1)
    train_loss.attach(trainer, loss_name)
    
    attach_metrics(trainer, chosen_diseases, "acc", Accuracy, True)
    attach_metrics(trainer, chosen_diseases, "prec", Precision, True)
    attach_metrics(trainer, chosen_diseases, "recall", Recall, True)
    attach_metrics(trainer, chosen_diseases, "roc_auc", utilsT.RocAucMetric, False)
    attach_metrics(trainer, chosen_diseases, "cm", ConfusionMatrix, get_transform_fn=get_transform_cm, metric_args=(2,))
    attach_metrics(trainer, chosen_diseases, "positives", RunningAverage, get_transform_fn=get_count_positives)
    

    timer = Timer(average=True)
    timer.attach(trainer, start=Events.EPOCH_STARTED, step=Events.EPOCH_COMPLETED)


    # Metrics callbacks
    if log_metrics is None:
        log_metrics = list(ALL_METRICS)

    def _write_metrics(run_type, metrics, epoch, wall_time):
        loss = metrics.get(loss_name, 0)

        writer.add_scalar("Loss/" + run_type, loss, epoch, wall_time)

        for metric_base_name in log_metrics:
            for disease in chosen_diseases:
                metric_value = metrics.get("{}_{}".format(metric_base_name, disease), -1)
                writer.add_scalar("{}_{}/{}".format(metric_base_name, disease, run_type), metric_value, epoch, wall_time)

    @trainer.on(Events.EPOCH_COMPLETED)
    def tb_write_metrics(trainer):
        epoch = trainer.state.epoch
        max_epochs = trainer.state.max_epochs

        # Run on evaluation
        validator.run(val_dataloader, 1)

        # Common time
        wall_time = time.time()

        # Log all metrics to TB
        _write_metrics("train", trainer.state.metrics, epoch, wall_time)
        _write_metrics("val", validator.state.metrics, epoch, wall_time)

        train_loss = trainer.state.metrics.get(loss_name, 0)
        val_loss = validator.state.metrics.get(loss_name, 0)

        tb_write_histogram(writer, model, epoch, wall_time)
        
        print("Finished epoch {}/{}, loss {} (val {})".format(epoch, max_epochs, train_loss, val_loss))

        
    # Hparam dict
    hparam_dict = {
        "resume": resume,
        "n_diseases": len(chosen_diseases),
        "diseases": ",".join(chosen_diseases),
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "opt": opt,
        "loss": loss_name,
        "samples (train, val)": "{},{}".format(train_samples, val_samples),
        "train_resnet": train_resnet,
    }
    
    def copy_params(params_dict, base_name):
        for name, value in params_dict.items():
            hparam_dict["{}_{}".format(base_name, name)] = value
    
    copy_params(loss_params, "loss")
    copy_params(opt_params, "opt")
    print("HPARAM: ", hparam_dict)


    # Train
    print("-" * 50)
    print("Training...")
    trainer.run(train_dataloader, n_epochs)
    

    # Capture time
    secs_per_epoch = timer.value()
    duration_per_epoch = utils.duration_to_str(int(secs_per_epoch))
    print("Average time per epoch: ", duration_per_epoch)
    print("-"*50)

    ## Write all hparams
    hparam_dict["duration_per_epoch"] = duration_per_epoch

    
    # FIXME: this is commented to avoid having too many hparams in TB frontend
    # metrics
#     def copy_metrics(engine, engine_name):
#         for metric_name, metric_value in engine.state.metrics.items():
#             hparam_dict["{}_{}".format(engine_name, metric_name)] = metric_value
#     copy_metrics(trainer, "train")
#     copy_metrics(validator, "val")

    print("Writing TB hparams")
    writer.add_hparams(hparam_dict, {})
    
    
    # Save model to disk
    if save:
        print("Saving model...")
        save_model(base_dir, run_name, experiment_mode, hparam_dict, trainer, model, optimizer)
    
    
    # Write graph to TB
    if write_graph:
        print("Writing TB graph...")
        tb_write_graph(writer, model, train_dataloader, device)


    # Write embeddings to TB
    if write_emb:
        print("Writing TB embeddings...")
        image_size = 256 if write_emb_img else 0
        
        # FIXME: be able to select images (balanced, train vs val, etc)
        image_list = list(train_dataset.label_index["FileName"])[:1000]
        # disease = chosen_diseases[0]
        # positive = train_dataset.label_index[train_dataset.label_index[disease] == 1]
        # negative = train_dataset.label_index[train_dataset.label_index[disease] == 0]
        # positive_images = list(positive["FileName"])[:25]
        # negative_images = list(negative["FileName"])[:25]
        # image_list = positive_images + negative_images

        all_images, all_embeddings, all_predictions, all_ground_truths = gen_embeddings(model,
                                                                                        train_dataset, 
                                                                                        device,
                                                                                        image_list=image_list,
                                                                                        image_size=image_size)
        tb_write_embeddings(writer,
                            chosen_diseases,
                            all_images, all_embeddings, all_predictions, all_ground_truths,
                            global_step=n_epochs,
                            use_images=write_emb_img,
                            tag="1000_{}".format("img" if write_emb_img else "no_img"),
                           )
        
    # Save confusion matrices (is expensive to calculate them afterwards)
    if save_cms:
        # REVIEW: delete this?
        # now is calculated with the ConfusionMatrix metric from ignite

        print("Saving confusion matrices...")
        # Assure folder
        cms_dir = os.path.join(base_dir, "cms", experiment_mode)
        os.makedirs(cms_dir, exist_ok=True)
        base_fname = os.path.join(cms_dir, run_name)
        
        n_diseases = len(chosen_diseases)

        def extract_cms(metrics):
            """Extract confusion matrices from a metrics dict."""
            cms = []
            for disease in chosen_diseases:
                key = "cm_" + disease
                if key not in metrics:
                    cm = np.array([[-1, -1], [-1, -1]])
                else:
                    cm = metrics[key].numpy()
                    
                cms.append(cm)
            return np.array(cms)
        
        # Train confusion matrix
        # n_train_images = train_dataset.size()[0]
        # all_results_train = utils.predict_all(model, train_dataloader, device, n_train_images, n_diseases)
        # train_cms = utils.calculate_all_cms(*all_results_train)
        train_cms = extract_cms(trainer.state.metrics)

        np.save(base_fname + "_train", train_cms)
        tb_write_cms(writer, "train", chosen_diseases, train_cms)
        
        # Validation confusion matrix
        # n_val_images = val_dataset.size()[0]
        # all_results_val = utils.predict_all(model, val_dataloader, device, n_val_images, n_diseases)
        # val_cms = utils.calculate_all_cms(*all_results_val)
        val_cms = extract_cms(validator.state.metrics)

        np.save(base_fname + "_val", val_cms)
        tb_write_cms(writer, "val", chosen_diseases, val_cms)
        
        # All confusion matrix
        all_cms = train_cms + val_cms
        np.save(base_fname + "_all", all_cms)
        
        # Print to console
        if len(chosen_diseases) == 1:
            print("Train CM: ")
            print(train_cms[0])
            print("Val CM: ")
            print(val_cms[0])
            
#             print("Train CM 2: ")
#             print(trainer.state.metrics["cm_" + chosen_diseases[0]])
#             print("Val CM 2: ")
#             print(validator.state.metrics["cm_" + chosen_diseases[0]])
        
    # Close TB writer
    if experiment_mode != "debug":
        writer.close()


    # Return values for debugging
    model_run = ModelRun(model, run_name, chosen_diseases)
    if experiment_mode == "debug":
        model_run.save_debug_data(writer, trainer, validator, train_dataset, train_dataloader, val_dataset, val_dataloader)
    
    return model_run

    
def parse_args():
    parser = argparse.ArgumentParser(description="Train a model", usage="%(prog)s [options]")
    
    parser.add_argument("--name", default="", type=str, help="Additional name to the run")
    parser.add_argument("--resume", default="", type=str, help="If present, resume from another run")
    parser.add_argument("--base-dir", default="/mnt/data/chest-x-ray-8", type=str, help="Base folder")
    parser.add_argument("--diseases", default=None, nargs="*", type=str, choices=utils.ALL_DISEASES,
                        help="Diseases to train with")
    parser.add_argument("--n-diseases", default=None, type=int, help="If present, select the first n diseases")
    parser.add_argument("--epochs", default=5, type=int, help="Amount of epochs")
    parser.add_argument("-bs", "--batch-size", default=4, type=int, help="Batch size")
    parser.add_argument("--shuffle", default=False, action="store_true",
                        help="If present, shuffles the train data at every epoch")
    parser.add_argument("--resnet", "--train-resnet", default=False, action="store_true",
                        help="Whether to retrain resnet layers or not")
    parser.add_argument("--image-format", default="RGB", type=str, help="Image format passed to Pillow")
    parser.add_argument("--metrics", "--log-metrics", default=None, nargs="*", choices=ALL_METRICS,
                        help="Metrics logged to TB")
    parser.add_argument("--flush", "--flush-secs", default=120, type=int, help="Tensorboard flush seconds")
    parser.add_argument("--train-images", default=None, type=int, help="Max train images")
    parser.add_argument("--val-images", default=None, type=int, help="Max validation images")
    parser.add_argument("--non-debug", default=False, action="store_true",
                        help="If present, is considered an official model")
    parser.add_argument("--tb-graph", default=False, action="store_true", help="If present save graph to TB")
    parser.add_argument("--tb-emb", default=False, action="store_true", help="If present save embedding to TB")
    parser.add_argument("--tb-emb-img", default=False, action="store_true", help="If present save embedding with images")
    
    loss_group = parser.add_argument_group(title="Losses options")
    loss_group.add_argument("--loss", default="wbce", type=str, choices=losses.AVAILABLE_LOSSES,
                            help="Loss function used")
    loss_group.add_argument("--focal-alpha", default=0.75, type=float, help="Alpha passed to focal loss")
    loss_group.add_argument("--focal-gamma", default=2, type=float, help="Gamma passed to focal loss")
    
    optim_group = parser.add_argument_group(title="Optimizer options")
    optim_group.add_argument("--opt", default="sgd", type=str, choices=optimizers.AVAILABLE_OPTIMIZERS,
                             help="Choose an optimizer")
    optim_group.add_argument("-lr", "--learning-rate", default=1e-6, type=float, help="Learning rate")
    optim_group.add_argument("-mom", "--momentum", default=0.9, type=float, help="Momentum passed to SGD")
    optim_group.add_argument("-wd", "--weight-decay", default=0, type=float, help="Weight decay passed to SGD/Adam")

    
    args = parser.parse_args()
    
    # If debugging, automatically set other values
    is_debug = not args.non_debug
    if is_debug:
        args.flush = 10
        args.train_images = args.train_images or 100
        args.val_images = args.val_images or 100
        
    # Prepare loss_params
    if args.loss == "focal_loss":
        args.loss_params = {
            "alpha": args.focal_alpha,
            "gamma": args.focal_gamma,
        }
    else:
        args.loss_params = {}
        
    # Prepare optim_params
    if args.opt == "sgd":
        args.opt_params = {
            "lr": args.learning_rate,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
        }
    elif args.opt == "adam":
        args.opt_params = {
            "lr": args.learning_rate,
            "weight_decay": args.weight_decay,
        }

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

    run = train_model(name=args.name,
                      resume=args.resume,
                      base_dir=args.base_dir,
                      chosen_diseases=args.diseases,
                      n_epochs=args.epochs,
                      batch_size=args.batch_size,
                      shuffle=args.shuffle,
                      opt=args.opt,
                      opt_params=args.opt_params,
                      loss_name=args.loss,
                      loss_params=args.loss_params,
                      train_resnet=args.resnet,
                      log_metrics=args.metrics,
                      flush_secs=args.flush,
                      train_max_images=args.train_images,
                      val_max_images=args.val_images,
                      experiment_mode=experiment_mode,
                      write_graph=args.tb_graph,
                      write_emb=args.tb_emb,
                      write_emb_img=args.tb_emb_img,
                      image_format=args.image_format,
                     )
    end_time = time.time()
    
    print("-"*50)
    print("Total training time: ", utils.duration_to_str(end_time - start_time))
    print("Run name: ", run.run_name)
    print("="*50)