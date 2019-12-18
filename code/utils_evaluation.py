"""
DeepSlide
Functions for evaluation.

Authors: Jason Wei, Behnaz Abdollahi, Saeed Hassanpour
"""

import operator
from pathlib import Path
from typing import (Dict, List, Tuple)

import numpy as np
from PIL import Image
from imageio import (imread, imsave)
from sklearn.metrics import confusion_matrix

from utils import (create_labels, get_all_image_paths, get_csv_paths)

Image.MAX_IMAGE_PIXELS = None

###########################################
#         THRESHOLD GRID SEARCH           #
###########################################


def get_prediction(patches_pred_file: Path, conf_thresholds: Dict[str, float],
                   image_ext: str) -> str:
    """
    Find the predicted class for a single WSI.

    Args:
        patches_pred_file: File containing the predicted classes for the patches that make up the WSI.
        conf_thresholds: Confidence thresholds to determine membership in a class (filter out noise).
        image_ext: Image extension for saving patches.

    Returns:
        A string containing the accuracy of classification for each class using the thresholds.
    """
    classes = list(conf_thresholds.keys())
    # Predicted class distribution per slide.
    class_to_count = {_class: 0 for _class in classes}

    # Looping through all the lines in the file and adding predictions.
    with patches_pred_file.open(mode="r") as patches_pred:

        patches_pred_lines = patches_pred.readlines()[1:]

        for line in patches_pred_lines:
            line_items = line[:-1].split(",")
            line_class = line_items[2]
            line_conf = float(line_items[3])
            if line_class in classes and line_conf > conf_thresholds[
                    line_class]:
                class_to_count[line_class] += 1
        if sum(class_to_count.values()) > 0:
            class_to_percent = {
                _class: class_to_count[_class] / sum(class_to_count.values())
                for _class in class_to_count
            }
        else:
            class_to_percent = {_class: 0 for _class in class_to_count}

    # Creating the line for output to CSV.
    return f"{Path(patches_pred_file.name).with_suffix(f'.{image_ext}')}," \
           f"{max(class_to_percent.items(), key=operator.itemgetter(1))[0]}," \
           f"{','.join([f'{class_to_percent[_class]:.5f}' for _class in classes])}," \
           f"{','.join([f'{class_to_count[_class]:.5f}' for _class in classes])}"


def output_all_predictions(patches_pred_folder: Path, output_folder: Path,
                           conf_thresholds: Dict[str, float],
                           classes: List[str], image_ext: str) -> None:
    """
    Output the predictions for the WSI into a CSV file.

    Args:
        patches_pred_folder: Folder containing the predicted classes for each patch.
        output_folder: Folder to save the predicted classes for each WSI for each threshold.
        conf_thresholds: The confidence thresholds for determining membership in a class (filter out noise).
        classes: Names of the classes in the dataset.
        image_ext: Image extension for saving patches.
    """
    # Open a new CSV file for each set of confidence thresholds used on each set of WSI.
    output_file = "".join([
        f"{_class}{str(conf_thresholds[_class])[1:]}_"
        for _class in conf_thresholds
    ])

    output_csv_path = output_folder.joinpath(f"{output_file[:-1]}.csv")

    # Confirm the output directory exists.
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to the output CSV.
    with output_csv_path.open(mode="w") as writer:
        writer.write(
            f"img,predicted,"
            f"{','.join([f'percent_{_class}' for _class in classes])},"
            f"{','.join([f'count_{_class}' for _class in classes])}\n")

        csv_paths = get_csv_paths(folder=patches_pred_folder)
        for csv_path in csv_paths:
            writer.write(
                f"{get_prediction(patches_pred_file=csv_path, conf_thresholds=conf_thresholds, image_ext=image_ext)}\n"
            )


def grid_search(pred_folder: Path, inference_folder: Path, classes: List[str],
                threshold_search: Tuple[float], image_ext: str) -> None:
    """
    Main function for performing the grid search over the confidence thresholds. Initially outputs
    predictions for each threshold.

    Args:
        pred_folder: Path containing the predictions.
        inference_folder: Path to write predictions to.
        classes: Names of the classes in the dataset.
        threshold_search: Threshold values to search.
        image_ext: Image extension for saving patches.
    """
    for threshold in threshold_search:
        output_all_predictions(
            patches_pred_folder=pred_folder,
            output_folder=inference_folder,
            conf_thresholds={_class: threshold
                             for _class in classes},
            classes=classes,
            image_ext=image_ext)


###########################################
#        FINDING BEST THRESHOLDS          #
###########################################
def get_scores(gt_labels: Dict[str, str], prediction_labels: Dict[str, str],
               classes: List[str]) -> Tuple[float, np.ndarray]:
    """
    Find the average class accuracy of the predictions.

    Args:
        gt_labels: Ground truth label dictionary from filenames to label strings.
        prediction_labels: Predicted label dictionary from filenames to label strings.
        classes: Names of the classes in the dataset.

    Returns:
        A tuple containing the average class accuracy and a confusion matrix.
    """
    class_to_gt_count = {_class: 0 for _class in classes}
    class_to_pred_count = {_class: 0 for _class in classes}
    gts = []
    preds = []

    for file in sorted(gt_labels.keys()):
        # Temporary fix. Need not to make folders with no crops.
        try:
            gt_label = gt_labels[file]
            pred_label = prediction_labels[file]
            gts.append(gt_label)
            preds.append(pred_label)

            # Predicted label is correct.
            if gt_label == pred_label:
                class_to_pred_count[gt_label] += 1

            # Add to total.
            class_to_gt_count[gt_label] += 1
        except KeyError:
            print(
                "WARNING: One of the image directories is empty. Skipping this directory."
            )
            continue

    conf_matrix = confusion_matrix(y_true=gts, y_pred=preds)
    class_to_acc = {
        _class: float(class_to_pred_count[_class]) / class_to_gt_count[_class]
        for _class in class_to_gt_count
    }
    avg_class_acc = sum(list(class_to_acc.values())) / len(class_to_acc)
    return avg_class_acc, conf_matrix


def parse_thresholds(csv_path: Path) -> Dict[str, float]:
    """
    Parse the CSV filename to find the classes for each threshold.

    Args:
        csv_path: Path to the CSV file containing the classes.

    Returns:
        A dictionary mapping class names to thresholds.
    """
    class_to_threshold = {}
    items = str(Path(csv_path.name).with_suffix("")).split("_")

    for item in items:
        subitems = item.split(".")
        _class = subitems[0]
        class_to_threshold[_class] = float(f"0.{subitems[1]}")

    return class_to_threshold


def find_best_acc_and_thresh(labels_csv: Path,
                             inference_folder: Path, classes: List[str]) -> \
        Dict[str, float]:
    """
    Find the best accuracy and threshold for the given images.

    Args:
        labels_csv: CSV file containing the ground truth labels.
        inference_folder: Folder containing the predicted labels. 
        classes: Names of the classes in the dataset.

    Returns:
        A dictionary mapping class names to the best thresholds.
    """
    gt_labels = create_labels(csv_path=labels_csv)
    prediction_csv_paths = get_csv_paths(folder=inference_folder)
    best_acc = 0
    best_thresholds = None
    best_csv = None
    for prediction_csv_path in prediction_csv_paths:
        prediction_labels = create_labels(csv_path=prediction_csv_path)
        avg_class_acc, conf_matrix = get_scores(
            gt_labels=gt_labels,
            prediction_labels=prediction_labels,
            classes=classes)
        print(f"thresholds {parse_thresholds(csv_path=prediction_csv_path)} "
              f"has average class accuracy {avg_class_acc:.5f}")
        if best_acc < avg_class_acc:
            best_acc = avg_class_acc
            best_csv = prediction_csv_path
            best_thresholds = parse_thresholds(csv_path=prediction_csv_path)
    print(f"view these predictions in {best_csv}")
    return best_thresholds


def print_final_test_results(labels_csv: Path, inference_folder: Path,
                             classes: List[str]) -> None:
    """
    Print the final accuracy and confusion matrix.

    Args:
        labels_csv: CSV file containing the ground truth labels.
        inference_folder: Folder containing the predicted labels.
        classes: Names of the classes in the dataset.
    """
    gt_labels = create_labels(csv_path=labels_csv)
    prediction_csv_paths = get_csv_paths(folder=inference_folder)
    for prediction_csv_path in prediction_csv_paths:
        prediction_labels = create_labels(csv_path=prediction_csv_path)
        avg_class_acc, conf_matrix = get_scores(
            gt_labels=gt_labels,
            prediction_labels=prediction_labels,
            classes=classes)
        print(f"test set has final avg class acc: {avg_class_acc:.5f}"
              f"\n{conf_matrix}")


###########################################
#             VISUALIZATION               #
###########################################
def color_to_np_color(color: str) -> np.ndarray:
    """
    Convert strings to NumPy colors.

    Args:
        color: The desired color as a string.

    Returns:
        The NumPy ndarray representation of the color.
    """
    colors = {
        "white": np.array([255, 255, 255]),
        "pink": np.array([255, 108, 180]),
        "black": np.array([0, 0, 0]),
        "red": np.array([255, 0, 0]),
        "purple": np.array([225, 225, 0]),
        "yellow": np.array([255, 255, 0]),
        "orange": np.array([255, 127, 80]),
        "blue": np.array([0, 0, 255]),
        "green": np.array([0, 255, 0])
    }
    return colors[color]


def add_predictions_to_image(
        xy_to_pred_class: Dict[Tuple[str, str], Tuple[str, float]],
        image: np.ndarray, prediction_to_color: Dict[str, np.ndarray],
        patch_size: int) -> np.ndarray:
    """
    Overlay the predicted dots (classes) on the WSI.

    Args:
        xy_to_pred_class: Dictionary mapping coordinates to predicted class along with the confidence.
        image: WSI to add predicted dots to.
        prediction_to_color: Dictionary mapping string color to NumPy ndarray color.
        patch_size: Size of the patches extracted from the WSI.

    Returns:
        The WSI with the predicted class dots overlaid.
    """
    for x, y in xy_to_pred_class.keys():
        prediction, __ = xy_to_pred_class[x, y]
        x = int(x)
        y = int(y)

        # Enlarge the dots so they are visible at larger scale.
        start = round((0.9 * patch_size) / 2)
        end = round((1.1 * patch_size) / 2)
        image[x + start:x + end, y + start:y +
              end, :] = prediction_to_color[prediction]

    return image


def get_xy_to_pred_class(window_prediction_folder: Path, img_name: str
                         ) -> Dict[Tuple[str, str], Tuple[str, float]]:
    """
    Find the dictionary of predictions.

    Args:
        window_prediction_folder: Path to the folder containing a CSV file with the predicted classes.
        img_name: Name of the image to find the predicted classes for.

    Returns:
        A dictionary mapping image coordinates to the predicted class and the confidence of the prediction.
    """
    xy_to_pred_class = {}

    with window_prediction_folder.joinpath(img_name).with_suffix(".csv").open(
            mode="r") as csv_lines_open:
        csv_lines = csv_lines_open.readlines()[1:]

        predictions = [line[:-1].split(",") for line in csv_lines]
        for prediction in predictions:
            x = prediction[0]
            y = prediction[1]
            pred_class = prediction[2]
            confidence = float(prediction[3])
            # Implement thresholding.
            xy_to_pred_class[(x, y)] = (pred_class, confidence)
    return xy_to_pred_class


def visualize(wsi_folder: Path, preds_folder: Path, vis_folder: Path,
              classes: List[str], num_classes: int, colors: Tuple[str],
              patch_size: int) -> None:
    """
    Main function for visualization.

    Args:
        wsi_folder: Path to WSI.
        preds_folder: Path containing the predicted classes.
        vis_folder: Path to output the WSI with overlaid classes to.
        classes: Names of the classes in the dataset.
        num_classes: Number of classes in the dataset.
        colors: Colors to use for visualization.
        patch_size: Size of the patches extracted from the WSI.
    """
    # Find list of WSI.
    whole_slides = get_all_image_paths(master_folder=wsi_folder)
    print(f"{len(whole_slides)} whole slides found from {wsi_folder}")
    prediction_to_color = {
        classes[i]: color_to_np_color(color=colors[i])
        for i in range(num_classes)
    }
    # Go over all of the WSI.
    for whole_slide in whole_slides:
        # Read in the image.
        whole_slide_numpy = imread(uri=whole_slide)[..., [0, 1, 2]]
        print(f"visualizing {whole_slide} "
              f"of shape {whole_slide_numpy.shape}")

        assert whole_slide_numpy.shape[
            2] == 3, f"Expected 3 channels while your image has {whole_slide_numpy.shape[2]} channels."

        # Save it.
        output_path = Path(
            f"{vis_folder.joinpath(whole_slide.name).with_suffix('')}"
            f"_predictions.jpg")

        # Confirm the output directory exists.
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Temporary fix. Need not to make folders with no crops.
        try:
            # Add the predictions to the image and save it.
            imsave(uri=output_path,
                   im=add_predictions_to_image(
                       xy_to_pred_class=get_xy_to_pred_class(
                           window_prediction_folder=preds_folder,
                           img_name=whole_slide.name),
                       image=whole_slide_numpy,
                       prediction_to_color=prediction_to_color,
                       patch_size=patch_size))
        except FileNotFoundError:
            print(
                "WARNING: One of the image directories is empty. Skipping this directory"
            )
            continue

    print(f"find the visualizations in {vis_folder}")
