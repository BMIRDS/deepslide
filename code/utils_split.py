"""
DeepSlide
Splits the data into training, validation, and testing sets.

Authors: Jason Wei, Behnaz Abdollahi, Saeed Hassanpour
"""

import shutil
from pathlib import Path
from typing import (Dict, List)

from utils import (get_image_paths, get_subfolder_paths)


def split(keep_orig_copy: bool, wsi_train: Path, wsi_val: Path, wsi_test: Path,
          classes: List[str], all_wsi: Path, val_wsi_per_class: int,
          test_wsi_per_class: int, labels_train: Path, labels_test: Path,
          labels_val: Path) -> None:
    """
    Main function for splitting data. Note that we want the
    validation and test sets to be balanced.

    Args:
        keep_orig_copy: Whether to move or copy the WSI when splitting into training, validation, and test sets.
        wsi_train: Location to be created to store WSI for training.
        wsi_val: Location to be created to store WSI for validation.
        wsi_test: Location to be created to store WSI for testing.
        classes: Names of the classes in the dataset.
        all_wsi: Location of the WSI organized in subfolders by class.
        val_wsi_per_class: Number of WSI per class to use in the validation set.
        test_wsi_per_class: Number of WSI per class to use in the test set.
        labels_train: Location to store the CSV file labels for training.
        labels_test: Location to store the CSV file labels for testing.
        labels_val: Location to store the CSV file labels for validation.
    """
    # Based on whether we want to move or keep the files.
    head = shutil.copyfile if keep_orig_copy else shutil.move

    # Create folders.
    for f in (wsi_train, wsi_val, wsi_test):
        subfolders = [f.joinpath(_class) for _class in classes]

        for subfolder in subfolders:
            # Confirm the output directory exists.
            subfolder.mkdir(parents=True, exist_ok=True)

    train_img_to_label = {}
    val_img_to_label = {}
    test_img_to_label = {}

    def move_set(folder: Path, image_files: List[Path],
                 ops: shutil) -> Dict[Path, str]:
        """
        Moves the sets to the desired output directories.

        Args:
            folder: Folder to move images to.
            image_files: Image files to move.
            ops: Whether to move or copy the files.

        Return:
            A dictionary mapping image filenames to classes.
        """
        def remove_topdir(filepath: Path) -> Path:
            """
            Remove the top directory since the filepath needs to be
            a relative path (i.e., a/b/c.jpg -> b/c.jpg).

            Args:
                filepath: Path to remove top directory from.

            Returns:
                Path with top directory removed.
            """
            return Path(*filepath.parts[1:])

        img_to_label = {}
        for image_file in image_files:
            # Copy or move the files.
            ops(src=image_file,
                dst=folder.joinpath(remove_topdir(filepath=image_file)))

            img_to_label[Path(image_file.name)] = image_file.parent.name

        return img_to_label

    # Sort the images and move/copy them appropriately.
    subfolder_paths = get_subfolder_paths(folder=all_wsi)
    for subfolder in subfolder_paths:
        image_paths = get_image_paths(folder=subfolder)

        # Make sure we have enough slides in each class.
        assert len(
            image_paths
        ) > val_wsi_per_class + test_wsi_per_class, "Not enough slides in each class."

        # Assign training, test, and validation images.
        test_idx = len(image_paths) - test_wsi_per_class
        val_idx = test_idx - val_wsi_per_class
        train_images = image_paths[:val_idx]
        val_images = image_paths[val_idx:test_idx]
        test_images = image_paths[test_idx:]
        print(f"class {Path(subfolder).name} "
              f"#train={len(train_images)} "
              f"#val={len(val_images)} "
              f"#test={len(test_images)}")

        # Move the training images.
        train_img_to_label.update(
            move_set(folder=wsi_train, image_files=train_images, ops=head))

        # Move the validation images.
        val_img_to_label.update(
            move_set(folder=wsi_val, image_files=val_images, ops=head))

        # Move the testing images.
        test_img_to_label.update(
            move_set(folder=wsi_test, image_files=test_images, ops=head))

    def write_to_csv(dest_filename: Path,
                     image_label_dict: Dict[Path, str]) -> None:
        """
        Write the image names and corresponding labels to a CSV file.

        Args:
            dest_filename: Destination filename for the CSV file.
            image_label_dict: Dictionary mapping filenames to labels.
        """
        with dest_filename.open(mode="w") as writer:
            writer.write("img,gt\n")
            for img in sorted(image_label_dict.keys()):
                writer.write(f"{img},{image_label_dict[img]}\n")

    write_to_csv(dest_filename=labels_train,
                 image_label_dict=train_img_to_label)
    write_to_csv(dest_filename=labels_val, image_label_dict=val_img_to_label)
    write_to_csv(dest_filename=labels_test, image_label_dict=test_img_to_label)
