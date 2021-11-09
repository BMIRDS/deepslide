"""
DeepSlide
General helper methods used in other functions.

Authors: Jason Wei, Behnaz Abdollahi, Saeed Hassanpour
"""

import datetime
from pathlib import Path
from typing import (Dict, List)

# Valid image extensions.
IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".svs", ".tif", ".tiff"]


def get_classes(folder: Path) -> List[str]:
    """
    Find the classes for classification.

    Args:
        folder: Folder containing the subfolders named by class.

    Returns:
        A list of strings corresponding to the class names.
    """
    return sorted([f.name for f in folder.iterdir() if
                   ((folder.joinpath(f.name).is_dir()) and (".DS_Store" not in f.name))], key=str)


def get_log_csv_name(log_folder: Path) -> Path:
    """
    Find the name of the CSV file for logging.

    Args:
        log_folder: Folder to save logging CSV file in.

    Returns:
        The path including the filename of the logging CSV file with date information.
    """
    now = datetime.datetime.now()

    return log_folder.joinpath(f"log_{now.month}{now.day}{now.year}"
                               f"_{now.hour}{now.minute}{now.second}.csv")


def get_image_names(folder: Path) -> List[Path]:
    """
    Find the names and paths of all of the images in a folder.

    Args:
        folder: Folder containing images (assume folder only contains images).

    Returns:
        A list of the names with paths of the images in a folder.
    """
    return sorted([Path(f.name) for f in folder.iterdir() if
                   ((folder.joinpath(f.name).is_file()) and (".DS_Store" not in f.name) and (f.suffix.casefold() in IMAGE_EXTS))], key=str)


def get_image_paths(folder: Path) -> List[Path]:
    """
    Find the full paths of the images in a folder.

    Args:
        folder: Folder containing images (assume folder only contains images).

    Returns:
        A list of the full paths to the images in the folder.
    """
    return sorted([folder.joinpath(f.name) for f in folder.iterdir() if
                   ((folder.joinpath(f.name).is_file()) and (".DS_Store" not in f.name) and (f.suffix.casefold() in IMAGE_EXTS))], key=str)


def get_subfolder_paths(folder: Path) -> List[Path]:
    """
    Find the paths of subfolders.

    Args:
        folder: Folder to look for subfolders in.

    Returns:
        A list containing the paths of the subfolders.
    """
    return sorted([folder.joinpath(f.name) for f in folder.iterdir() if
                   ((folder.joinpath(f.name).is_dir()) and (".DS_Store" not in f.name))], key=str)


def get_all_image_paths(master_folder: Path) -> List[Path]:
    """
    Finds all image paths in subfolders.

    Args:
        master_folder: Root folder containing subfolders.

    Returns:
        A list of the paths to the images found in the folder.
    """
    all_paths = []
    subfolders = get_subfolder_paths(folder=master_folder)
    if len(subfolders) > 1:
        for subfolder in subfolders:
            all_paths += get_image_paths(folder=subfolder)
    else:
        all_paths = get_image_paths(folder=master_folder)
    return all_paths


def get_csv_paths(folder: Path) -> List[Path]:
    """
    Find the CSV files contained in a folder.

    Args:
        folder: Folder to search for CSV files.

    Returns:
        A list of the paths to the CSV files in the folder.
    """
    return sorted([folder.joinpath(f.name) for f in folder.iterdir() if (
                (folder.joinpath(f.name).is_file()) and ("csv" in f.name) and (".DS_Store" not in f.name))],
                  key=str)


def create_labels(csv_path: Path) -> Dict[str, str]:
    """
    Read the labels from a CSV file.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        A dictionary mapping string filenames to string labels.
    """
    with csv_path.open(mode="r") as lines_open:
        lines = lines_open.readlines()[1:]

        file_to_gt_label = {}

        for line in lines:
            if len(line) > 3:
                pieces = line[:-1].split(",")
                file = pieces[0]
                gt_label = pieces[1]
                file_to_gt_label[file] = gt_label

    return file_to_gt_label
