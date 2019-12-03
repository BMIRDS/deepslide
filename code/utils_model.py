"""
DeepSlide
Using ResNet to train and test.

Authors: Jason Wei, Behnaz Abdollahi, Saeed Hassanpour, Naofumi Tomita

Last Modified: November 30, 2019 (Joseph DiPalma)
"""

import operator
import random
import time
from pathlib import Path
from typing import (Dict, IO, List)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from PIL import Image
from torch.optim import lr_scheduler
from torchvision import (datasets, transforms)

import config
from utils import (get_image_paths, get_subfolder_paths)

###########################################
#             MISC FUNCTIONS              #
###########################################


def calculate_confusion_matrix(all_labels: np.ndarray,
                               all_predicts: np.ndarray) -> None:
    """
    Prints the confusion matrix from the given data.

    Args:
        all_labels: The ground truth labels.
        all_predicts: The predicted labels.
    """
    remap_classes = {x: config.classes[x] for x in range(config.num_classes)}

    # Set print options.
    # Sources:
    #   1. https://stackoverflow.com/questions/42735541/customized-float-formatting-in-a-pandas-dataframe
    #   2. https://stackoverflow.com/questions/11707586/how-do-i-expand-the-output-display-to-see-more-columns-of-a-pandas-dataframe
    #   3. https://pandas.pydata.org/pandas-docs/stable/user_guide/style.html
    pd.options.display.float_format = "{:.2f}".format
    pd.options.display.width = 0

    cm = pd.crosstab(index=pd.Series(pd.Categorical(
        pd.Series(all_labels).replace(remap_classes),
        categories=config.classes),
                                     name="Actual"),
                     columns=pd.Series(pd.Categorical(
                         pd.Series(all_predicts).replace(remap_classes),
                         categories=config.classes),
                                       name="Predicted"),
                     normalize="index")

    cm.style.hide_index()
    print(cm)


class Random90Rotation:
    def __init__(self, degrees: List[int] = None) -> None:
        """
        Randomly rotate the image for training. Credits to Naofumi Tomita.

        Args:
            degrees: Degrees available for rotation.
        """
        self.degrees = [0, 90, 180, 270] if (degrees is None) else degrees

    def __call__(self, im: Image) -> Image:
        """
        Produces a randomly rotated image every time the instance is called.

        Args:
            im: The image to rotate.

        Returns:    
            Randomly rotated image.
        """
        return im.rotate(angle=random.sample(population=self.degrees, k=1)[0])


def create_model() -> torchvision.models.resnet.ResNet:
    """
    Instantiate the ResNet model.

    Returns:
        The instantiated ResNet model with the requested parameters.
    """
    assert config.args.num_layers in [
        18, 34, 50, 101, 152
    ], f"Invalid number of ResNet Layers.  " \
       f"Must be one of [18, 34, 50, 101, 152] and not {config.args.num_layers}"
    model_constructor = getattr(torchvision.models,
                                f"resnet{config.args.num_layers}")
    model = model_constructor(num_classes=config.num_classes)

    if config.args.pretrain:
        pretrained = model_constructor(pretrained=True).state_dict()
        if config.num_classes != pretrained["fc.weight"].size(0):
            del pretrained["fc.weight"], pretrained["fc.bias"]
        model.load_state_dict(state_dict=pretrained, strict=False)
    return model


def get_data_transforms() -> Dict[str, torchvision.transforms.Compose]:
    """
    Sets up the dataset transforms for training and validation.

    Returns:
        A dictionary mapping training and validation strings to data transforms.
    """
    return {
        "train":
        transforms.Compose(transforms=[
            transforms.ColorJitter(
                brightness=config.args.color_jitter_brightness,
                contrast=config.args.color_jitter_contrast,
                saturation=config.args.color_jitter_saturation,
                hue=config.args.color_jitter_hue),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            Random90Rotation(),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.path_mean, std=config.path_std)
        ]),
        "val":
        transforms.Compose(transforms=[
            transforms.ToTensor(),
            transforms.Normalize(mean=config.path_mean, std=config.path_std)
        ])
    }


def print_params() -> None:
    """
    Print the configuration of the model.
    """
    print(f"train_folder: {config.args.train_folder}\n"
          f"num_epochs: {config.args.num_epochs}\n"
          f"num_layers: {config.args.num_layers}\n"
          f"learning_rate: {config.args.learning_rate}\n"
          f"batch_size: {config.args.batch_size}\n"
          f"weight_decay: {config.args.weight_decay}\n"
          f"learning_rate_decay: {config.args.learning_rate_decay}\n"
          f"resume_checkpoint: {config.args.resume_checkpoint}\n"
          f"resume_checkpoint_path (only if resume_checkpoint is true): "
          f"{config.resume_checkpoint_path}\n"
          f"save_interval: {config.args.save_interval}\n"
          f"output in checkpoints_folder: {config.args.checkpoints_folder}\n"
          f"pretrain: {config.args.pretrain}\n"
          f"log_csv: {config.log_csv}\n\n")


###########################################
#          MAIN TRAIN FUNCTION            #
###########################################


def train_helper(model: torchvision.models.resnet.ResNet,
                 dataloaders: Dict[str, torch.utils.data.DataLoader],
                 dataset_sizes: Dict[str, int],
                 criterion: torch.nn.modules.loss, optimizer: torch.optim,
                 scheduler: torch.optim.lr_scheduler, num_epochs: int,
                 writer: IO) -> None:
    """
    Function for training ResNet.

    Args:
        model: ResNet model for training.
        dataloaders: Dataloaders for IO pipeline.
        dataset_sizes: Sizes of the training and validation dataset.
        criterion: Metric used for calculating loss.
        optimizer: Optimizer to use for gradient descent.
        scheduler: Scheduler to use for learning rate decay.
        num_epochs: Number of epochs to train the network.
        writer: Writer to write logging information.
    """
    since = time.time()

    # Initialize all the tensors to be used in training and validation.
    # Do this outside the loop since it will be written over entirely at each
    # epoch and doesn't need to be reallocated each time.
    train_all_labels = torch.empty(size=(dataset_sizes["train"], ),
                                   dtype=torch.long,
                                   device=config.device)
    train_all_predicts = torch.empty(size=(dataset_sizes["train"], ),
                                     dtype=torch.long,
                                     device=config.device)
    val_all_labels = torch.empty(size=(dataset_sizes["val"], ),
                                 dtype=torch.long,
                                 device=config.device)
    val_all_predicts = torch.empty(size=(dataset_sizes["val"], ),
                                   dtype=torch.long,
                                   device=config.device)

    # Train for specified number of epochs.
    for epoch in range(num_epochs, config.args.num_epochs):

        # Training phase.
        model.train(mode=True)

        train_running_loss = 0.0
        train_running_corrects = 0

        # Train over all training data.
        for idx, (inputs, labels) in enumerate(dataloaders["train"]):
            train_inputs = inputs.to(device=config.device)
            train_labels = labels.to(device=config.device)
            optimizer.zero_grad()

            # Forward and backpropagation.
            with torch.set_grad_enabled(mode=True):
                train_outputs = model(train_inputs)
                __, train_preds = torch.max(train_outputs, dim=1)
                train_loss = criterion(input=train_outputs,
                                       target=train_labels)
                train_loss.backward()
                optimizer.step()

            # Update training diagnostics.
            train_running_loss += train_loss.item() * train_inputs.size(0)
            train_running_corrects += torch.sum(
                train_preds == train_labels.data, dtype=torch.double)

            train_all_labels[idx * config.args.batch_size:(idx + 1) *
                             config.args.batch_size] = train_labels.data
            train_all_predicts[idx * config.args.batch_size:(idx + 1) *
                               config.args.batch_size] = train_preds

        calculate_confusion_matrix(
            all_labels=train_all_labels.cpu().numpy(),
            all_predicts=train_all_predicts.cpu().numpy())

        # Store training diagnostics.
        train_loss = train_running_loss / dataset_sizes["train"]
        train_acc = train_running_corrects / dataset_sizes["train"]

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Validation phase.
        model.train(mode=False)

        val_running_loss = 0.0
        val_running_corrects = 0

        # Feed forward over all the validation data.
        for idx, (val_inputs, val_labels) in enumerate(dataloaders["val"]):
            val_inputs = val_inputs.to(device=config.device)
            val_labels = val_labels.to(device=config.device)

            # Feed forward.
            with torch.set_grad_enabled(mode=False):
                val_outputs = model(val_inputs)
                _, val_preds = torch.max(val_outputs, dim=1)
                val_loss = criterion(input=val_outputs, target=val_labels)

            # Update validation diagnostics.
            val_running_loss += val_loss.item() * val_inputs.size(0)
            val_running_corrects += torch.sum(val_preds == val_labels.data,
                                              dtype=torch.double)

            val_all_labels[idx * config.args.batch_size:(idx + 1) *
                           config.args.batch_size] = val_labels.data
            val_all_predicts[idx * config.args.batch_size:(idx + 1) *
                             config.args.batch_size] = val_preds

        calculate_confusion_matrix(all_labels=val_all_labels.cpu().numpy(),
                                   all_predicts=val_all_predicts.cpu().numpy())

        # Store validation diagnostics.
        val_loss = val_running_loss / dataset_sizes["val"]
        val_acc = val_running_corrects / dataset_sizes["val"]

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        scheduler.step()

        current_lr = None
        for group in optimizer.param_groups:
            current_lr = group["lr"]

        # Remaining things related to training.
        if epoch % config.args.save_interval == 0:
            epoch_output_path = config.args.checkpoints_folder.joinpath(
                f"resnet{config.args.num_layers}_e{epoch}_va{val_acc:.5f}.pt")

            # Confirm the output directory exists.
            epoch_output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save the model as a state dictionary.
            torch.save(obj={
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch + 1
            },
                       f=str(epoch_output_path))

        writer.write(f"{epoch},{train_loss:.4f},"
                     f"{train_acc:.4f},{val_loss:.4f},{val_acc:.4f}")

        # Print the diagnostics for each epoch.
        print(f"Epoch {epoch} with lr "
              f"{current_lr:.15f}: "
              f"t_loss: {train_loss:.4f} "
              f"t_acc: {train_acc:.4f} "
              f"v_loss: {val_loss:.4f} "
              f"v_acc: {val_acc:.4f}\n")

    # Print training information at the end.
    print(f"\ntraining complete in "
          f"{(time.time() - since) // 60:.2f} minutes")


def train_resnet() -> None:
    """
    Main function for training ResNet.
    """
    # Loading in the data.
    data_transforms = get_data_transforms()

    image_datasets = {
        x: datasets.ImageFolder(root=str(config.args.train_folder.joinpath(x)),
                                transform=data_transforms[x])
        for x in ["train", "val"]
    }

    dataloaders = {
        x: torch.utils.data.DataLoader(dataset=image_datasets[x],
                                       batch_size=config.args.batch_size,
                                       shuffle=(x is "train"),
                                       num_workers=config.args.num_workers)
        for x in ["train", "val"]
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}

    print(
        f"{len(config.classes)} classes: {config.classes}\n"
        f"num train images {len(dataloaders['train']) * config.args.batch_size}\n"
        f"num val images {len(dataloaders['val']) * config.args.batch_size}\n"
        f"CUDA is_available: {torch.cuda.is_available()}")

    model = create_model()
    model = model.to(device=config.device)
    optimizer = optim.Adam(params=model.parameters(),
                           lr=config.args.learning_rate,
                           weight_decay=config.args.weight_decay)
    scheduler = lr_scheduler.ExponentialLR(
        optimizer=optimizer, gamma=config.args.learning_rate_decay)

    # Initialize the model.
    if config.args.resume_checkpoint:
        ckpt = torch.load(f=config.resume_checkpoint_path)
        model.load_state_dict(state_dict=ckpt["model_state_dict"])
        optimizer.load_state_dict(state_dict=ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(state_dict=ckpt["scheduler_state_dict"])
        epoch = ckpt["epoch"]
        print(f"model loaded from {config.resume_checkpoint_path}")
    else:
        epoch = 0

    # Print the model hyperparameters.
    print_params()

    # Logging the model after every epoch.
    # Confirm the output directory exists.
    config.log_csv.parent.mkdir(parents=True, exist_ok=True)

    with config.log_csv.open(mode="w") as writer:
        writer.write("epoch,train_loss,train_acc,val_loss,val_acc\n")
        # Train the model.
        train_helper(model=model,
                     dataloaders=dataloaders,
                     dataset_sizes=dataset_sizes,
                     criterion=nn.CrossEntropyLoss(),
                     optimizer=optimizer,
                     scheduler=scheduler,
                     num_epochs=epoch,
                     writer=writer)


###########################################
#      MAIN EVALUATION FUNCTION           #
###########################################


def parse_val_acc(model_path: Path) -> float:
    """
    Parse the validation accuracy from the filename.

    Args:
        model_path: The model path to parse for the validation accuracy.

    Returns:
        The parsed validation accuracy.
    """
    return float(
        f"{('.'.join(model_path.name.split('.')[:-1])).split('_')[-1][2:]}")


def get_best_model(checkpoints_folder: Path) -> str:
    """
    Finds the model with the best validation accuracy.

    Args:
        checkpoints_folder: Folder containing the models to test.

    Returns:
        The location of the model with the best validation accuracy.
    """
    return max({
        model: parse_val_acc(model_path=model)
        for model in get_image_paths(folder=checkpoints_folder)
    }.items(),
               key=operator.itemgetter(1))[0]


def get_predictions(patches_eval_folder: Path, output_folder: Path) -> None:
    """
    Main function for running the model on all of the generated patches.

    Args:
        patches_eval_folder: Folder containing patches to evaluate on.
        output_folder: Folder to save the model results to.
    """
    # Initialize the model.
    model_path = get_best_model(
        checkpoints_folder=config.args.checkpoints_folder
    ) if config.args.auto_select else config.eval_model

    model = create_model()
    ckpt = torch.load(f=model_path)
    model.load_state_dict(state_dict=ckpt["model_state_dict"])
    model = model.to(device=config.device)

    model.train(mode=False)
    print(f"model loaded from {model_path}")

    # For outputting the predictions.
    class_num_to_class = {
        i: config.classes[i]
        for i in range(len(config.classes))
    }

    start = time.time()
    # Load the data for each folder.
    image_folders = get_subfolder_paths(folder=patches_eval_folder)

    # Where we want to write out the predictions.
    # Confirm the output directory exists.
    output_folder.mkdir(parents=True, exist_ok=True)

    # For each WSI.
    for image_folder in image_folders:

        # Load the image dataset.
        dataloader = torch.utils.data.DataLoader(
            dataset=datasets.ImageFolder(
                root=str(image_folder),
                transform=transforms.Compose(transforms=[
                    transforms.ToTensor(),
                    transforms.Normalize(mean=config.path_mean,
                                         std=config.path_std)
                ])),
            batch_size=config.args.batch_size,
            shuffle=False,
            num_workers=config.args.num_workers)
        num_test_image_windows = len(dataloader) * config.args.batch_size

        # Load the image names so we know the coordinates of the patches we are predicting.
        image_folder = image_folder.joinpath(image_folder.name)
        window_names = get_image_paths(folder=image_folder)

        print(f"testing on {num_test_image_windows} crops from {image_folder}")

        with output_folder.joinpath(f"{image_folder.name}.csv").open(
                mode="w") as writer:

            writer.write("x,y,prediction,confidence\n")

            # Loop through all of the patches.
            for batch_num, (test_inputs, test_labels) in enumerate(dataloader):
                batch_window_names = window_names[
                    batch_num *
                    config.args.batch_size:batch_num * config.args.batch_size +
                    config.args.batch_size]

                confidences, test_preds = torch.max(nn.Softmax(dim=1)(model(
                    test_inputs.to(device=config.device))),
                                                    dim=1)
                for i in range(test_preds.shape[0]):
                    # Find coordinates and predicted class.
                    xy = batch_window_names[i].name.split(".")[0].split(";")

                    writer.write(
                        f"{','.join([xy[0], xy[1], f'{class_num_to_class[test_preds[i].data.item()]}', f'{confidences[i].data.item():.5f}'])}\n"
                    )

    print(f"time for {patches_eval_folder}: {time.time() - start:.2f} seconds")
