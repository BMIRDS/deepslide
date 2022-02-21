"""
DeepSlide
Using ResNet to train and test.

Authors: Jason Wei, Behnaz Abdollahi, Saeed Hassanpour, Naofumi Tomita
"""

import operator
import random
import time
from pathlib import Path
from typing import (Dict, IO, List, Tuple)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from PIL import Image
from torch.optim import lr_scheduler
from torchvision import (datasets, transforms)

from utils import (get_image_paths, get_subfolder_paths)

###########################################
#             MISC FUNCTIONS              #
###########################################


def calculate_confusion_matrix(all_labels: np.ndarray,
                               all_predicts: np.ndarray, classes: List[str],
                               num_classes: int) -> None:
    """
    Prints the confusion matrix from the given data.

    Args:
        all_labels: The ground truth labels.
        all_predicts: The predicted labels.
        classes: Names of the classes in the dataset.
        num_classes: Number of classes in the dataset.
    """
    remap_classes = {x: classes[x] for x in range(num_classes)}

    # Set print options.
    # Sources:
    #   1. https://stackoverflow.com/questions/42735541/customized-float-formatting-in-a-pandas-dataframe
    #   2. https://stackoverflow.com/questions/11707586/how-do-i-expand-the-output-display-to-see-more-columns-of-a-pandas-dataframe
    #   3. https://pandas.pydata.org/pandas-docs/stable/user_guide/style.html
    pd.options.display.float_format = "{:.2f}".format
    pd.options.display.width = 0

    actual = pd.Series(pd.Categorical(
        pd.Series(all_labels).replace(remap_classes), categories=classes),
                       name="Actual")

    predicted = pd.Series(pd.Categorical(
        pd.Series(all_predicts).replace(remap_classes), categories=classes),
                          name="Predicted")

    cm = pd.crosstab(index=actual, columns=predicted, normalize="index", dropna=False)

    cm.style.hide_index()
    print(cm)


class Random90Rotation:
    def __init__(self, degrees: Tuple[int] = None) -> None:
        """
        Randomly rotate the image for training. Credits to Naofumi Tomita.

        Args:
            degrees: Degrees available for rotation.
        """
        self.degrees = (0, 90, 180, 270) if (degrees is None) else degrees

    def __call__(self, im: Image) -> Image:
        """
        Produces a randomly rotated image every time the instance is called.

        Args:
            im: The image to rotate.

        Returns:    
            Randomly rotated image.
        """
        return im.rotate(angle=random.sample(population=self.degrees, k=1)[0])


def create_model(num_layers: int, num_classes: int,
                 pretrain: bool) -> torchvision.models.resnet.ResNet:
    """
    Instantiate the ResNet model.

    Args:
        num_layers: Number of layers to use in the ResNet model from [18, 34, 50, 101, 152].
        num_classes: Number of classes in the dataset.
        pretrain: Use pretrained ResNet weights.

    Returns:
        The instantiated ResNet model with the requested parameters.
    """
    assert num_layers in (
        18, 34, 50, 101, 152
    ), f"Invalid number of ResNet Layers. Must be one of [18, 34, 50, 101, 152] and not {num_layers}"
    model_constructor = getattr(torchvision.models, f"resnet{num_layers}")
    model = model_constructor(num_classes=num_classes)

    if pretrain:
        pretrained = model_constructor(pretrained=True).state_dict()
        if num_classes != pretrained["fc.weight"].size(0):
            del pretrained["fc.weight"], pretrained["fc.bias"]
        model.load_state_dict(state_dict=pretrained, strict=False)
    return model


def get_data_transforms(color_jitter_brightness: float,
                        color_jitter_contrast: float,
                        color_jitter_saturation: float,
                        color_jitter_hue: float, path_mean: List[float],
                        path_std: List[float]
                        ) -> Dict[str, torchvision.transforms.Compose]:
    """
    Sets up the dataset transforms for training and validation.

    Args:
        color_jitter_brightness: Random brightness jitter to use in data augmentation for ColorJitter() transform.
        color_jitter_contrast: Random contrast jitter to use in data augmentation for ColorJitter() transform.
        color_jitter_saturation: Random saturation jitter to use in data augmentation for ColorJitter() transform.
        color_jitter_hue: Random hue jitter to use in data augmentation for ColorJitter() transform.
        path_mean: Means of the WSIs for each dimension.
        path_std: Standard deviations of the WSIs for each dimension.

    Returns:
        A dictionary mapping training and validation strings to data transforms.
    """
    return {
        "train":
        transforms.Compose(transforms=[
            transforms.ColorJitter(brightness=color_jitter_brightness,
                                   contrast=color_jitter_contrast,
                                   saturation=color_jitter_saturation,
                                   hue=color_jitter_hue),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            Random90Rotation(),
            transforms.ToTensor(),
            transforms.Normalize(mean=path_mean, std=path_std)
        ]),
        "val":
        transforms.Compose(transforms=[
            transforms.ToTensor(),
            transforms.Normalize(mean=path_mean, std=path_std)
        ])
    }


def print_params(train_folder: Path, num_epochs: int, num_layers: int,
                 learning_rate: float, batch_size: int, weight_decay: float,
                 learning_rate_decay: float, resume_checkpoint: bool,
                 resume_checkpoint_path: Path, save_interval: int,
                 checkpoints_folder: Path, pretrain: bool,
                 log_csv: Path) -> None:
    """
    Print the configuration of the model.

    Args:
        train_folder: Location of the automatically built training input folder.
        num_epochs: Number of epochs for training.
        num_layers: Number of layers to use in the ResNet model from [18, 34, 50, 101, 152].
        learning_rate: Learning rate to use for gradient descent.
        batch_size: Mini-batch size to use for training.
        weight_decay: Weight decay (L2 penalty) to use in optimizer.
        learning_rate_decay: Learning rate decay amount per epoch.
        resume_checkpoint: Resume model from checkpoint file.
        resume_checkpoint_path: Path to the checkpoint file for resuming training.
        save_interval: Number of epochs between saving checkpoints.
        checkpoints_folder: Directory to save model checkpoints to.
        pretrain: Use pretrained ResNet weights.
        log_csv: Name of the CSV file containing the logs.
    """
    print(f"train_folder: {train_folder}\n"
          f"num_epochs: {num_epochs}\n"
          f"num_layers: {num_layers}\n"
          f"learning_rate: {learning_rate}\n"
          f"batch_size: {batch_size}\n"
          f"weight_decay: {weight_decay}\n"
          f"learning_rate_decay: {learning_rate_decay}\n"
          f"resume_checkpoint: {resume_checkpoint}\n"
          f"resume_checkpoint_path (only if resume_checkpoint is true): "
          f"{resume_checkpoint_path}\n"
          f"save_interval: {save_interval}\n"
          f"output in checkpoints_folder: {checkpoints_folder}\n"
          f"pretrain: {pretrain}\n"
          f"log_csv: {log_csv}\n\n")


###########################################
#          MAIN TRAIN FUNCTION            #
###########################################


def train_helper(model: torchvision.models.resnet.ResNet,
                 dataloaders: Dict[str, torch.utils.data.DataLoader],
                 dataset_sizes: Dict[str, int],
                 criterion: torch.nn.modules.loss, optimizer: torch.optim,
                 scheduler: torch.optim.lr_scheduler, num_epochs: int,
                 writer: IO, device: torch.device, start_epoch: int,
                 batch_size: int, save_interval: int, checkpoints_folder: Path,
                 num_layers: int, classes: List[str],
                 num_classes: int) -> None:
    """
    Function for training ResNet.

    Args:
        model: ResNet model for training.
        dataloaders: Dataloaders for IO pipeline.
        dataset_sizes: Sizes of the training and validation dataset.
        criterion: Metric used for calculating loss.
        optimizer: Optimizer to use for gradient descent.
        scheduler: Scheduler to use for learning rate decay.
        start_epoch: Starting epoch for training.
        writer: Writer to write logging information.
        device: Device to use for running model.
        num_epochs: Total number of epochs to train for.
        batch_size: Mini-batch size to use for training.
        save_interval: Number of epochs between saving checkpoints.
        checkpoints_folder: Directory to save model checkpoints to.
        num_layers: Number of layers to use in the ResNet model from [18, 34, 50, 101, 152].
        classes: Names of the classes in the dataset.
        num_classes: Number of classes in the dataset.
    """
    since = time.time()

    # Initialize all the tensors to be used in training and validation.
    # Do this outside the loop since it will be written over entirely at each
    # epoch and doesn't need to be reallocated each time.
    train_all_labels = torch.empty(size=(dataset_sizes["train"], ),
                                   dtype=torch.long).cpu()
    train_all_predicts = torch.empty(size=(dataset_sizes["train"], ),
                                     dtype=torch.long).cpu()
    val_all_labels = torch.empty(size=(dataset_sizes["val"], ),
                                 dtype=torch.long).cpu()
    val_all_predicts = torch.empty(size=(dataset_sizes["val"], ),
                                   dtype=torch.long).cpu()

    # Train for specified number of epochs.
    for epoch in range(start_epoch, num_epochs):

        # Training phase.
        model.train(mode=True)

        train_running_loss = 0.0
        train_running_corrects = 0

        # Train over all training data.
        for idx, (inputs, labels) in enumerate(dataloaders["train"]):
            train_inputs = inputs.to(device=device)
            train_labels = labels.to(device=device)
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

            start = idx * batch_size
            end = start + batch_size

            train_all_labels[start:end] = train_labels.detach().cpu()
            train_all_predicts[start:end] = train_preds.detach().cpu()

        calculate_confusion_matrix(all_labels=train_all_labels.numpy(),
                                   all_predicts=train_all_predicts.numpy(),
                                   classes=classes,
                                   num_classes=num_classes)

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
            val_inputs = val_inputs.to(device=device)
            val_labels = val_labels.to(device=device)

            # Feed forward.
            with torch.set_grad_enabled(mode=False):
                val_outputs = model(val_inputs)
                _, val_preds = torch.max(val_outputs, dim=1)
                val_loss = criterion(input=val_outputs, target=val_labels)

            # Update validation diagnostics.
            val_running_loss += val_loss.item() * val_inputs.size(0)
            val_running_corrects += torch.sum(val_preds == val_labels.data,
                                              dtype=torch.double)

            start = idx * batch_size
            end = start + batch_size

            val_all_labels[start:end] = val_labels.detach().cpu()
            val_all_predicts[start:end] = val_preds.detach().cpu()

        calculate_confusion_matrix(all_labels=val_all_labels.numpy(),
                                   all_predicts=val_all_predicts.numpy(),
                                   classes=classes,
                                   num_classes=num_classes)

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
        if epoch % save_interval == 0:
            epoch_output_path = checkpoints_folder.joinpath(
                f"resnet{num_layers}_e{epoch}_va{val_acc:.5f}.pt")

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
                     f"{train_acc:.4f},{val_loss:.4f},{val_acc:.4f}\n")

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


def train_resnet(
        train_folder: Path, batch_size: int, num_workers: int,
        device: torch.device, classes: List[str], learning_rate: float,
        weight_decay: float, learning_rate_decay: float,
        resume_checkpoint: bool, resume_checkpoint_path: Path, log_csv: Path,
        color_jitter_brightness: float, color_jitter_contrast: float,
        color_jitter_hue: float, color_jitter_saturation: float,
        path_mean: List[float], path_std: List[float], num_classes: int,
        num_layers: int, pretrain: bool, checkpoints_folder: Path,
        num_epochs: int, save_interval: int) -> None:
    """
    Main function for training ResNet.

    Args:
        train_folder: Location of the automatically built training input folder.
        batch_size: Mini-batch size to use for training.
        num_workers: Number of workers to use for IO.
        device: Device to use for running model.
        classes: Names of the classes in the dataset.
        learning_rate: Learning rate to use for gradient descent.
        weight_decay: Weight decay (L2 penalty) to use in optimizer.
        learning_rate_decay: Learning rate decay amount per epoch.
        resume_checkpoint: Resume model from checkpoint file.
        resume_checkpoint_path: Path to the checkpoint file for resuming training.
        log_csv: Name of the CSV file containing the logs.
        color_jitter_brightness: Random brightness jitter to use in data augmentation for ColorJitter() transform.
        color_jitter_contrast: Random contrast jitter to use in data augmentation for ColorJitter() transform.
        color_jitter_hue: Random hue jitter to use in data augmentation for ColorJitter() transform.
        color_jitter_saturation: Random saturation jitter to use in data augmentation for ColorJitter() transform.
        path_mean: Means of the WSIs for each dimension.
        path_std: Standard deviations of the WSIs for each dimension.
        num_classes: Number of classes in the dataset.
        num_layers: Number of layers to use in the ResNet model from [18, 34, 50, 101, 152].
        pretrain: Use pretrained ResNet weights.
        checkpoints_folder: Directory to save model checkpoints to.
        num_epochs: Number of epochs for training.
        save_interval: Number of epochs between saving checkpoints.
    """
    # Loading in the data.
    data_transforms = get_data_transforms(
        color_jitter_brightness=color_jitter_brightness,
        color_jitter_contrast=color_jitter_contrast,
        color_jitter_hue=color_jitter_hue,
        color_jitter_saturation=color_jitter_saturation,
        path_mean=path_mean,
        path_std=path_std)

    image_datasets = {
        x: datasets.ImageFolder(root=str(train_folder.joinpath(x)),
                                transform=data_transforms[x])
        for x in ("train", "val")
    }

    dataloaders = {
        x: torch.utils.data.DataLoader(dataset=image_datasets[x],
                                       batch_size=batch_size,
                                       shuffle=(x is "train"),
                                       num_workers=num_workers)
        for x in ("train", "val")
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ("train", "val")}

    print(f"{num_classes} classes: {classes}\n"
          f"num train images {len(dataloaders['train']) * batch_size}\n"
          f"num val images {len(dataloaders['val']) * batch_size}\n"
          f"CUDA is_available: {torch.cuda.is_available()}")

    model = create_model(num_classes=num_classes,
                         num_layers=num_layers,
                         pretrain=pretrain)
    model = model.to(device=device)
    optimizer = optim.Adam(params=model.parameters(),
                           lr=learning_rate,
                           weight_decay=weight_decay)
    scheduler = lr_scheduler.ExponentialLR(optimizer=optimizer,
                                           gamma=learning_rate_decay)

    # Initialize the model.
    if resume_checkpoint:
        ckpt = torch.load(f=resume_checkpoint_path)
        model.load_state_dict(state_dict=ckpt["model_state_dict"])
        optimizer.load_state_dict(state_dict=ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(state_dict=ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"]
        print(f"model loaded from {resume_checkpoint_path}")
    else:
        start_epoch = 0

    # Print the model hyperparameters.
    print_params(batch_size=batch_size,
                 checkpoints_folder=checkpoints_folder,
                 learning_rate=learning_rate,
                 learning_rate_decay=learning_rate_decay,
                 log_csv=log_csv,
                 num_epochs=num_epochs,
                 num_layers=num_layers,
                 pretrain=pretrain,
                 resume_checkpoint=resume_checkpoint,
                 resume_checkpoint_path=resume_checkpoint_path,
                 save_interval=save_interval,
                 train_folder=train_folder,
                 weight_decay=weight_decay)

    # Logging the model after every epoch.
    # Confirm the output directory exists.
    log_csv.parent.mkdir(parents=True, exist_ok=True)

    with log_csv.open(mode="w") as writer:
        writer.write("epoch,train_loss,train_acc,val_loss,val_acc\n")
        # Train the model.
        train_helper(model=model,
                     dataloaders=dataloaders,
                     dataset_sizes=dataset_sizes,
                     criterion=nn.CrossEntropyLoss(),
                     optimizer=optimizer,
                     scheduler=scheduler,
                     start_epoch=start_epoch,
                     writer=writer,
                     batch_size=batch_size,
                     checkpoints_folder=checkpoints_folder,
                     device=device,
                     num_layers=num_layers,
                     save_interval=save_interval,
                     num_epochs=num_epochs,
                     classes=classes,
                     num_classes=num_classes)


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
        for model in [m for m in checkpoints_folder.rglob("*.pt") if ".DS_Store" not in str(m)]
    }.items(),
               key=operator.itemgetter(1))[0]


def get_predictions(patches_eval_folder: Path, output_folder: Path,
                    checkpoints_folder: Path, auto_select: bool,
                    eval_model: Path, device: torch.device, classes: List[str],
                    num_classes: int, path_mean: List[float],
                    path_std: List[float], num_layers: int, pretrain: bool,
                    batch_size: int, num_workers: int) -> None:
    """
    Main function for running the model on all of the generated patches.

    Args:
        patches_eval_folder: Folder containing patches to evaluate on.
        output_folder: Folder to save the model results to.
        checkpoints_folder: Directory to save model checkpoints to.
        auto_select: Automatically select the model with the highest validation accuracy,
        eval_model: Path to the model with the highest validation accuracy.
        device: Device to use for running model.
        classes: Names of the classes in the dataset.
        num_classes: Number of classes in the dataset.
        path_mean: Means of the WSIs for each dimension.
        path_std: Standard deviations of the WSIs for each dimension.
        num_layers: Number of layers to use in the ResNet model from [18, 34, 50, 101, 152].
        pretrain: Use pretrained ResNet weights.
        batch_size: Mini-batch size to use for training.
        num_workers: Number of workers to use for IO.
    """
    # Initialize the model.
    model_path = get_best_model(
        checkpoints_folder=checkpoints_folder) if auto_select else eval_model

    model = create_model(num_classes=num_classes,
                         num_layers=num_layers,
                         pretrain=pretrain)
    ckpt = torch.load(f=model_path)
    model.load_state_dict(state_dict=ckpt["model_state_dict"])
    model = model.to(device=device)

    model.train(mode=False)
    print(f"model loaded from {model_path}")

    # For outputting the predictions.
    class_num_to_class = {i: classes[i] for i in range(num_classes)}

    start = time.time()
    # Load the data for each folder.
    image_folders = get_subfolder_paths(folder=patches_eval_folder)

    # Where we want to write out the predictions.
    # Confirm the output directory exists.
    output_folder.mkdir(parents=True, exist_ok=True)

    # For each WSI.
    for image_folder in image_folders:

        # Temporary fix. Need not to make folders with no crops.
        try:
            # Load the image dataset.
            dataloader = torch.utils.data.DataLoader(
                dataset=datasets.ImageFolder(
                    root=str(image_folder),
                    transform=transforms.Compose(transforms=[
                        transforms.ToTensor(),
                        transforms.Normalize(mean=path_mean, std=path_std)
                    ])),
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers)
        except RuntimeError:
            print(
                "WARNING: One of the image directories is empty. Skipping this directory."
            )
            continue

        num_test_image_windows = len(dataloader) * batch_size

        # Load the image names so we know the coordinates of the patches we are predicting.
        image_folder = image_folder.joinpath(image_folder.name)
        window_names = get_image_paths(folder=image_folder)

        print(f"testing on {num_test_image_windows} crops from {image_folder}")

        with output_folder.joinpath(f"{image_folder.name}.csv").open(
                mode="w") as writer:

            writer.write("x,y,prediction,confidence\n")

            # Loop through all of the patches.
            for batch_num, (test_inputs, test_labels) in enumerate(dataloader):
                batch_window_names = window_names[batch_num *
                                                  batch_size:batch_num *
                                                  batch_size + batch_size]

                confidences, test_preds = torch.max(nn.Softmax(dim=1)(model(
                    test_inputs.to(device=device))),
                                                    dim=1)
                for i in range(test_preds.shape[0]):
                    # Find coordinates and predicted class.
                    xy = batch_window_names[i].name.split(".")[0].split(";")

                    writer.write(
                        f"{','.join([xy[0], xy[1], f'{class_num_to_class[test_preds[i].data.item()]}', f'{confidences[i].data.item():.5f}'])}\n"
                    )

    print(f"time for {patches_eval_folder}: {time.time() - start:.2f} seconds")
