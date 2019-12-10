"""
DeepSlide
Contains all hyperparameters for the entire repository.

Authors: Jason Wei, Behnaz Abdollahi, Saeed Hassanpour

Last Modified: November 30, 2019 (Joseph DiPalma)
"""

import argparse
from pathlib import Path

import torch

from compute_stats import compute_stats
from utils import (get_classes, get_log_csv_name)

# Source: https://stackoverflow.com/questions/12151306/argparse-way-to-include-default-values-in-help
parser = argparse.ArgumentParser(
    description="DeepSlide",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

###########################################
#               USER INPUTS               #
###########################################
# Input folders for training images.
# Must contain subfolders of images labelled by class.
# If your two classes are 'a' and 'n', you must have a/*.jpg with the images in class a and
# n/*.jpg with the images in class n.
parser.add_argument(
    "--all-wsi",
    type=Path,
    default=Path("all_wsi"),
    help="Location of the WSI organized in subfolders by class")
# For splitting into validation set.
parser.add_argument("--val-wsi-per-class",
                    type=int,
                    default=20,
                    help="Number of WSI per class to use in validation set")
# For splitting into testing set, remaining images used in train.
parser.add_argument("--test-wsi-per-class",
                    type=int,
                    default=30,
                    help="Number of WSI per class to use in test set")
# When splitting, do you want to move WSI or copy them?
parser.add_argument(
    "--keep-orig-copy",
    type=bool,
    default=True,
    help=
    "Whether to move or copy the WSI when splitting into training, validation, and test sets"
)

#######################################
#               GENERAL               #
#######################################
# Number of processes to use.
parser.add_argument("--num-workers",
                    type=int,
                    default=8,
                    help="Number of workers to use for IO")
# Default shape for ResNet in PyTorch.
parser.add_argument("--patch_size",
                    type=int,
                    default=224,
                    help="Size of the patches extracted from the WSI")

##########################################
#               DATA SPLIT               #
##########################################
# The names of your to-be folders.
parser.add_argument("--wsi-train",
                    type=Path,
                    default=Path("wsi_train"),
                    help="Location to be created to store WSI for training")
parser.add_argument("--wsi-val",
                    type=Path,
                    default=Path("wsi_val"),
                    help="Location to be created to store WSI for validation")
parser.add_argument("--wsi-test",
                    type=Path,
                    default=Path("wsi_test"),
                    help="Location to be created to store WSI for testing")

# Where the CSV file labels will go.
parser.add_argument("--labels-train",
                    type=Path,
                    default=Path("labels_train.csv"),
                    help="Location to store the CSV file labels for training")
parser.add_argument(
    "--labels-val",
    type=Path,
    default=Path("labels_val.csv"),
    help="Location to store the CSV file labels for validation")
parser.add_argument("--labels-test",
                    type=Path,
                    default=Path("labels_test.csv"),
                    help="Location to store the CSV file labels for testing")

###############################################################
#               PROCESSING AND PATCH GENERATION               #
###############################################################
# This is the input for model training, automatically built.
parser.add_argument(
    "--train-folder",
    type=Path,
    default=Path("train_folder"),
    help="Location of the automatically built training input folder")

# Folders of patches by WSI in training set, used for finding training accuracy at WSI level.
parser.add_argument(
    "--patches-eval-train",
    type=Path,
    default=Path("patches_eval_train"),
    help=
    "Folders of patches by WSI in training set, used for finding training accuracy at WSI level"
)
# Folders of patches by WSI in validation set, used for finding validation accuracy at WSI level.
parser.add_argument(
    "--patches-eval-val",
    type=Path,
    default=Path("patches_eval_val"),
    help=
    "Folders of patches by WSI in validation set, used for finding validation accuracy at WSI level"
)
# Folders of patches by WSI in test set, used for finding test accuracy at WSI level.
parser.add_argument(
    "--patches-eval-test",
    type=Path,
    default=Path("patches_eval_test"),
    help=
    "Folders of patches by WSI in testing set, used for finding test accuracy at WSI level"
)

# Target number of training patches per class.
parser.add_argument("--num-train-per-class",
                    type=int,
                    default=80000,
                    help="Target number of training samples per class")

# Only looks for purple images and filters whitespace.
parser.add_argument(
    "--type-histopath",
    type=bool,
    default=True,
    help="Only look for purple histopathology images and filter whitespace")

# Sliding window overlap factor (for testing).
# For generating patches during the training phase, we slide a window to overlap by some factor.
# Must be an integer. 1 means no overlap, 2 means overlap by 1/2, 3 means overlap by 1/3.
# Recommend 2 for very high resolution, 3 for medium, and 5 not extremely high resolution images.
parser.add_argument("--slide-overlap",
                    type=int,
                    default=3,
                    help="Sliding window overlap factor for the testing phase")

parser.add_argument("--image-ext",
                    type=str,
                    default="jpg",
                    help="Image extension for saving patches")

#########################################
#               TRANSFORM               #
#########################################
parser.add_argument(
    "--color-jitter-brightness",
    type=float,
    default=0.5,
    help=
    "Random brightness jitter to use in data augmentation for ColorJitter() transform"
)
parser.add_argument(
    "--color-jitter-contrast",
    type=float,
    default=0.5,
    help=
    "Random contrast jitter to use in data augmentation for ColorJitter() transform"
)
parser.add_argument(
    "--color-jitter-saturation",
    type=float,
    default=0.5,
    help=
    "Random saturation jitter to use in data augmentation for ColorJitter() transform"
)
parser.add_argument(
    "--color-jitter-hue",
    type=float,
    default=0.2,
    help=
    "Random hue jitter to use in data augmentation for ColorJitter() transform"
)

########################################
#               TRAINING               #
########################################
# Model hyperparameters.
parser.add_argument("--num-epochs",
                    type=int,
                    default=20,
                    help="Number of epochs for training")
# Choose from [18, 34, 50, 101, 152].
parser.add_argument(
    "--num-layers",
    type=int,
    default=18,
    help=
    "Number of layers to use in the ResNet model from [18, 34, 50, 101, 152]")
parser.add_argument("--learning-rate",
                    type=float,
                    default=0.001,
                    help="Learning rate to use for gradient descent")
parser.add_argument("--batch-size",
                    type=int,
                    default=16,
                    help="Mini-batch size to use for training")
parser.add_argument("--weight-decay",
                    type=float,
                    default=1e-4,
                    help="Weight decay (L2 penalty) to use in optimizer")
parser.add_argument("--learning-rate-decay",
                    type=float,
                    default=0.85,
                    help="Learning rate decay amount per epoch")
parser.add_argument("--resume-checkpoint",
                    type=bool,
                    default=False,
                    help="Resume model from checkpoint file")
parser.add_argument("--save-interval",
                    type=int,
                    default=1,
                    help="Number of epochs between saving checkpoints")
# Where models are saved.
parser.add_argument("--checkpoints-folder",
                    type=Path,
                    default=Path("checkpoints"),
                    help="Directory to save model checkpoints to")

# Name of checkpoint file to load from.
parser.add_argument(
    "--checkpoint-file",
    type=Path,
    default=Path("xyz.pt"),
    help="Checkpoint file to load if resume_checkpoint_path is True")
# ImageNet pretrain?
parser.add_argument("--pretrain",
                    type=bool,
                    default=False,
                    help="Use pretrained ResNet weights")
parser.add_argument("--log-folder",
                    type=Path,
                    default=Path("logs"),
                    help="Directory to save logs to")

##########################################
#               PREDICTION               #
##########################################
# Selecting the best model.
# Automatically select the model with the highest validation accuracy.
parser.add_argument(
    "--auto-select",
    type=bool,
    default=True,
    help="Automatically select the model with the highest validation accuracy")
# Where to put the training prediction CSV files.
parser.add_argument(
    "--preds-train",
    type=Path,
    default=Path("preds_train"),
    help="Directory for outputting training prediction CSV files")
# Where to put the validation prediction CSV files.
parser.add_argument(
    "--preds-val",
    type=Path,
    default=Path("preds_val"),
    help="Directory for outputting validation prediction CSV files")
# Where to put the testing prediction CSV files.
parser.add_argument(
    "--preds-test",
    type=Path,
    default=Path("preds_test"),
    help="Directory for outputting testing prediction CSV files")

##########################################
#               EVALUATION               #
##########################################
# Folder for outputting WSI predictions based on each threshold.
parser.add_argument(
    "--inference-train",
    type=Path,
    default=Path("inference_train"),
    help=
    "Folder for outputting WSI training predictions based on each threshold")
parser.add_argument(
    "--inference-val",
    type=Path,
    default=Path("inference_val"),
    help=
    "Folder for outputting WSI validation predictions based on each threshold")
parser.add_argument(
    "--inference-test",
    type=Path,
    default=Path("inference_test"),
    help="Folder for outputting WSI testing predictions based on each threshold"
)

# For visualization.
parser.add_argument(
    "--vis-train",
    type=Path,
    default=Path("vis_train"),
    help="Folder for outputting the WSI training prediction visualizations")
parser.add_argument(
    "--vis-val",
    type=Path,
    default=Path("vis_val"),
    help="Folder for outputting the WSI validation prediction visualizations")
parser.add_argument(
    "--vis-test",
    type=Path,
    default=Path("vis_test"),
    help="Folder for outputting the WSI testing prediction visualizations")

#######################################################
#               ARGUMENTS FROM ARGPARSE               #
#######################################################
args = parser.parse_args()

# Device to use for PyTorch code.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Automatically read in the classes.
classes = get_classes(folder=args.all_wsi)
num_classes = len(classes)

# This is the input for model training, automatically built.
train_patches = args.train_folder.joinpath("train")
val_patches = args.train_folder.joinpath("val")

# Compute the mean and standard deviation for the given set of WSI for normalization.
path_mean, path_std = compute_stats(folderpath=args.all_wsi)

# Only used is resume_checkpoint is True.
resume_checkpoint_path = args.checkpoints_folder.joinpath(args.checkpoint_file)

# Named with date and time.
log_csv = get_log_csv_name(log_folder=args.log_folder)

# Does nothing if auto_select is True.
eval_model = args.checkpoints_folder.joinpath(args.checkpoint_file)

# Find the best threshold for filtering noise (discard patches with a confidence less than this threshold).
threshold_search = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# For visualization.
# This order is the same order as your sorted classes.
colors = ("red", "white", "blue", "green", "purple", "orange", "black", "pink", "yellow")

# Print the configuration.
# Source: https://stackoverflow.com/questions/44689546/how-to-print-out-a-dictionary-nicely-in-python/44689627
print(f"\n\n\n###############     CONFIGURATION     ###############\n"
      f"{chr(10).join(f'{k}:{chr(9)}{v}' for k, v in vars(args).items())}"
      f"device:\t{device}\n"
      f"classes:\t{classes}\n"
      f"num_classes:\t{num_classes}\n"
      f"train_patches:\t{train_patches}\n"
      f"val_patches:\t{val_patches}\n"
      f"path_mean:\t{path_mean}\n"
      f"path_std:\t{path_std}\n"
      f"resume_checkpoint_path:\t{resume_checkpoint_path}\n"
      f"log_csv:\t{log_csv}\n"
      f"eval_model:\t{eval_model}\n"
      f"threshold_search:\t{threshold_search}\n"
      f"colors:\t{colors}\n"
      f"\n#####################################################\n\n\n")
