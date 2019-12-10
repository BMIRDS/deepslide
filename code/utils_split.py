"""
DeepSlide
Splits the data into training, validation, and testing sets.

Authors: Jason Wei, Behnaz Abdollahi, Saeed Hassanpour

Last Modified: November 30, 2019 (Joseph DiPalma)
"""

import shutil
from pathlib import Path
from typing import (Dict, List)

import config
from utils import (get_image_paths, get_subfolder_paths)


def split() -> None:
    """
    Main function for splitting data. Note that we want the
    validation and test sets to be balanced.
    """
    # Based on whether we want to move or keep the files.
    head = shutil.copyfile if config.args.keep_orig_copy else shutil.move

    # Create folders.
    for f in (config.args.wsi_train, config.args.wsi_val, config.args.wsi_test):
        subfolders = [f.joinpath(_class) for _class in config.classes]

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
    subfolder_paths = get_subfolder_paths(folder=config.args.all_wsi)
    for subfolder in subfolder_paths:
        image_paths = get_image_paths(folder=subfolder)

        # Make sure we have enough slides in each class.
        assert len(
            image_paths
        ) > config.args.val_wsi_per_class + config.args.test_wsi_per_class, "Not enough slides in each class."

        # Assign training, test, and validation images.
        test_idx = len(image_paths) - config.args.test_wsi_per_class
        val_idx = test_idx - config.args.val_wsi_per_class
        train_images = image_paths[:val_idx]
        val_images = image_paths[val_idx:test_idx]
        test_images = image_paths[test_idx:]
        print(f"class {Path(subfolder).name} "
              f"#train={len(train_images)} "
              f"#val={len(val_images)} "
              f"#test={len(test_images)}")

        # Move the training images.
        train_img_to_label.update(
            move_set(folder=config.args.wsi_train,
                     image_files=train_images,
                     ops=head))

        # Move the validation images.
        val_img_to_label.update(
            move_set(folder=config.args.wsi_val,
                     image_files=val_images,
                     ops=head))

        # Move the testing images.
        test_img_to_label.update(
            move_set(folder=config.args.wsi_test,
                     image_files=test_images,
                     ops=head))

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

    write_to_csv(dest_filename=config.args.labels_train,
                 image_label_dict=train_img_to_label)
    write_to_csv(dest_filename=config.args.labels_val,
                 image_label_dict=val_img_to_label)
    write_to_csv(dest_filename=config.args.labels_test,
                 image_label_dict=test_img_to_label)
