"""
DeepSlide
Contains all functions for processing.

Authors: Jason Wei, Behnaz Abdollahi, Saeed Hassanpour
"""

import functools
import itertools
import math
import time
from multiprocessing import (Process, Queue, RawArray)
from pathlib import Path
from shutil import copyfile
from typing import (Callable, Dict, List, Tuple)

import numpy as np
from PIL import Image
from imageio import (imsave, imread)
from skimage.measure import block_reduce

from utils import (get_all_image_paths, get_image_names, get_image_paths,
                   get_subfolder_paths)

Image.MAX_IMAGE_PIXELS = None


def is_purple(crop: np.ndarray, purple_threshold: int,
              purple_scale_size: int) -> bool:
    """
    Determines if a given portion of an image is purple.

    Args:
        crop: Portion of the image to check for being purple.
        purple_threshold: Number of purple points for region to be considered purple.
        purple_scale_size: Scalar to use for reducing image to check for purple.

    Returns:
        A boolean representing whether the image is purple or not.
    """
    block_size = (crop.shape[0] // purple_scale_size,
                  crop.shape[1] // purple_scale_size, 1)
    pooled = block_reduce(image=crop, block_size=block_size, func=np.average)

    # Calculate boolean arrays for determining if portion is purple.
    r, g, b = pooled[..., 0], pooled[..., 1], pooled[..., 2]
    cond1 = r > g - 10
    cond2 = b > g - 10
    cond3 = ((r + b) / 2) > g + 20

    # Find the indexes of pooled satisfying all 3 conditions.
    pooled = pooled[cond1 & cond2 & cond3]
    num_purple = pooled.shape[0]

    return num_purple > purple_threshold


###########################################
#         GENERATING TRAINING DATA        #
###########################################


def get_folder_size_and_num_images(folder: Path) -> Tuple[float, int]:
    """
    Finds the number and size of images in a folder path.
    Used to decide how much to slide windows.

    Args:
        folder: Folder containing images.

    Returns:
        A tuple containing the total size of the images and the number of images.
    """
    image_paths = get_image_paths(folder=folder)

    file_size = 0
    for image_path in image_paths:
        file_size += image_path.stat().st_size

    file_size_mb = file_size / 1e6
    return file_size_mb, len(image_paths)


def get_subfolder_to_overlap(subfolders: List[Path],
                             desired_crops_per_class: int
                             ) -> Dict[Path, float]:
    """
    Find how much the inverse overlap factor should be for each folder so that
    the class distributions are approximately equal.

    Args:
        subfolders: Subfolders to calculate the overlap factors for.
        desired_crops_per_class: Desired number of patches per class.

    Returns:
        A dictionary mapping subfolder paths to inverse overlap factor.
    """
    subfolder_to_overlap_factor = {}
    for subfolder in subfolders:
        subfolder_size, subfolder_num_images = get_folder_size_and_num_images(
            folder=subfolder)

        # Each image is 13KB = 0.013MB, idk I just added two randomly.
        overlap_factor = max(
            1.0,
            math.pow(
                math.sqrt(desired_crops_per_class / (subfolder_size / 0.013)),
                1.5))
        subfolder_to_overlap_factor[subfolder] = overlap_factor
        print(f"{subfolder}: {subfolder_size}MB, "
              f"{subfolder_num_images} images, "
              f"overlap_factor={overlap_factor:.2f}")

    return subfolder_to_overlap_factor


def gen_train_patches(input_folder: Path, output_folder: Path,
                      num_train_per_class: int, num_workers: int,
                      patch_size: int, purple_threshold: int,
                      purple_scale_size: int, image_ext: str,
                      type_histopath: bool) -> None:
    """
    Generates all patches for subfolders in the training set.

    Args:
        input_folder: Folder containing the subfolders containing WSI.
        output_folder: Folder to save the patches to.
        num_train_per_class: The desired number of training patches per class.
        num_workers: Number of workers to use for IO.
        patch_size: Size of the patches extracted from the WSI.
        purple_threshold: Number of purple points for region to be considered purple.
        purple_scale_size: Scalar to use for reducing image to check for purple.
        image_ext: Image extension for saving patches.
        type_histopath: Only look for purple histopathology images and filter whitespace.
    """
    # Find the subfolders and how much patches should overlap for each.
    subfolders = get_subfolder_paths(folder=input_folder)
    print(f"{subfolders} subfolders found from {input_folder}")
    subfolder_to_overlap_factor = get_subfolder_to_overlap(
        subfolders=subfolders, desired_crops_per_class=num_train_per_class)

    # Produce the patches.
    for input_subfolder in subfolders:
        produce_patches(input_folder=input_subfolder,
                        output_folder=output_folder.joinpath(
                            input_subfolder.name),
                        inverse_overlap_factor=subfolder_to_overlap_factor[
                            input_subfolder],
                        by_folder=False,
                        num_workers=num_workers,
                        patch_size=patch_size,
                        purple_threshold=purple_threshold,
                        purple_scale_size=purple_scale_size,
                        image_ext=image_ext,
                        type_histopath=type_histopath)

    print("\nfinished all folders\n")


def gen_val_patches(input_folder: Path, output_folder: Path,
                    overlap_factor: float, num_workers: int, patch_size: int,
                    purple_threshold: int, purple_scale_size: int,
                    image_ext: str, type_histopath: bool) -> None:
    """
    Generates all patches for subfolders in the validation set.

    Args:
        input_folder: Folder containing the subfolders containing WSI.
        output_folder: Folder to save the patches to.
        overlap_factor: The amount of overlap between patches.
        num_workers: Number of workers to use for IO.
        patch_size: Size of the patches extracted from the WSI.
        purple_threshold: Number of purple points for region to be considered purple.
        purple_scale_size: Scalar to use for reducing image to check for purple.
        image_ext: Image extension for saving patches.
        type_histopath: Only look for purple histopathology images and filter whitespace.
    """
    # Find the subfolders and how much patches should overlap for each.
    subfolders = get_subfolder_paths(folder=input_folder)
    print(f"{len(subfolders)} subfolders found from {input_folder}")

    # Produce the patches.
    for input_subfolder in subfolders:
        produce_patches(input_folder=input_subfolder,
                        output_folder=output_folder.joinpath(
                            input_subfolder.name),
                        inverse_overlap_factor=overlap_factor,
                        by_folder=False,
                        num_workers=num_workers,
                        patch_size=patch_size,
                        purple_threshold=purple_threshold,
                        purple_scale_size=purple_scale_size,
                        image_ext=image_ext,
                        type_histopath=type_histopath)

    print("\nfinished all folders\n")


###########################################
#       BALANCING CLASS DISTRIBUTION      #
###########################################


def duplicate_until_n(image_paths: List[Path], n: int) -> None:
    """
    Duplicate the underrepresented classes to balance class distributions.

    Args:
        image_paths: Image paths to check for balance.
        n: Desired number of images.
    """
    num_dupls = n - len(image_paths)

    print(f"balancing {image_paths[0].parent} by duplicating {num_dupls}")

    for i in range(num_dupls):
        image_path = image_paths[i % len(image_paths)]

        xys = image_path.name.split("_")
        x = xys[:-2]
        y = xys[-2:]

        copyfile(src=image_path,
                 dst=Path(
                     image_path.parent, f"{'_'.join(x)}dup"
                     f"{(i // len(image_paths)) + 2}_"
                     f"{'_'.join(y)}"))


def balance_classes(training_folder: Path) -> None:
    """
    Balancing class distribution so that training isn't skewed.

    Args:
        training_folder: Folder containing the subfolders to be balanced.
    """
    subfolders = get_subfolder_paths(folder=training_folder)
    subfolder_to_images = {
        subfolder: get_image_paths(folder=subfolder)
        for subfolder in subfolders
    }

    # Find the class with the most images.
    biggest_size = max({
        subfolder: len(subfolder_to_images[subfolder])
        for subfolder in subfolders
    }.values())

    for subfolder in subfolder_to_images:
        duplicate_until_n(image_paths=subfolder_to_images[subfolder],
                          n=biggest_size)

    print(f"balanced all training classes to have {biggest_size} images\n")


def find_patch_mp(func: Callable[[Tuple[int, int]], int], in_queue: Queue,
                  out_queue: Queue) -> None:
    """
    Find the patches from the WSI using multiprocessing.
    Helper function to ensure values are sent to each process
    correctly.

    Args:
        func: Function to call in multiprocessing.
        in_queue: Queue containing input data.
        out_queue: Queue to put output in.
    """
    while True:
        xy = in_queue.get()
        if xy is None:
            break
        out_queue.put(obj=func(xy))


def find_patch(xy_start: Tuple[int, int], output_folder: Path,
               image: np.ndarray, by_folder: bool, image_loc: Path,
               patch_size: int, image_ext: str, type_histopath: bool,
               purple_threshold: int, purple_scale_size: int) -> int:
    """
    Find the patches for a WSI.

    Args:
        output_folder: Folder to save the patches to.
        image: WSI to extract patches from.
        xy_start: Starting coordinates of the patch.
        by_folder: Whether to generate the patches by folder or by image.
        image_loc: Location of the image to use for creating output filename.
        patch_size: Size of the patches extracted from the WSI.
        image_ext: Image extension for saving patches.
        type_histopath: Only look for purple histopathology images and filter whitespace.
        purple_threshold: Number of purple points for region to be considered purple.
        purple_scale_size: Scalar to use for reducing image to check for purple.

    Returns:
        The number 1 if the image was saved successfully and a 0 otherwise.
        Used to determine the number of patches produced per WSI.
    """
    x_start, y_start = xy_start

    patch = image[x_start:x_start + patch_size, y_start:y_start +
                  patch_size, :]
    # Sometimes the images are RGBA instead of RGB. Only keep RGB channels.
    patch = patch[..., [0, 1, 2]]

    if by_folder:
        output_subsubfolder = output_folder.joinpath(
            Path(image_loc.name).with_suffix(""))
        output_subsubfolder = output_subsubfolder.joinpath(
            output_subsubfolder.name)
        output_subsubfolder.mkdir(parents=True, exist_ok=True)
        output_path = output_subsubfolder.joinpath(
            f"{str(x_start).zfill(5)};{str(y_start).zfill(5)}.{image_ext}")
    else:
        output_path = output_folder.joinpath(
            f"{image_loc.stem}_{x_start}_{y_start}.{image_ext}")

    if type_histopath:
        if is_purple(crop=patch,
                     purple_threshold=purple_threshold,
                     purple_scale_size=purple_scale_size):
            imsave(uri=output_path, im=patch)
        else:
            return 0
    else:
        imsave(uri=output_path, im=patch)
    return 1


def produce_patches(input_folder: Path, output_folder: Path,
                    inverse_overlap_factor: float, by_folder: bool,
                    num_workers: int, patch_size: int, purple_threshold: int,
                    purple_scale_size: int, image_ext: str,
                    type_histopath: bool) -> None:
    """
    Produce the patches from the WSI in parallel.

    Args:
        input_folder: Folder containing the WSI.
        output_folder: Folder to save the patches to.
        inverse_overlap_factor: Overlap factor used in patch creation.
        by_folder: Whether to generate the patches by folder or by image.
        num_workers: Number of workers to use for IO.
        patch_size: Size of the patches extracted from the WSI.
        purple_threshold: Number of purple points for region to be considered purple.
        purple_scale_size: Scalar to use for reducing image to check for purple.
        image_ext: Image extension for saving patches.
        type_histopath: Only look for purple histopathology images and filter whitespace.
    """
    output_folder.mkdir(parents=True, exist_ok=True)
    image_locs = get_all_image_paths(
        master_folder=input_folder) if by_folder else get_image_names(
            folder=input_folder)
    outputted_patches = 0

    print(f"\ngetting small crops from {len(image_locs)} "
          f"images in {input_folder} "
          f"with inverse overlap factor {inverse_overlap_factor:.2f} "
          f"outputting in {output_folder}")

    start_time = time.time()

    for image_loc in image_locs:
        image = imread(
            uri=(image_loc if by_folder else input_folder.joinpath(image_loc)))

        # Sources:
        # 1. https://research.wmz.ninja/articles/2018/03/on-sharing-large-arrays-when-using-pythons-multiprocessing.html
        # 2. https://stackoverflow.com/questions/33247262/the-corresponding-ctypes-type-of-a-numpy-dtype
        # 3. https://stackoverflow.com/questions/7894791/use-numpy-array-in-shared-memory-for-multiprocessing
        img = RawArray(
            typecode_or_type=np.ctypeslib.as_ctypes_type(dtype=image.dtype),
            size_or_initializer=image.size)
        img_np = np.frombuffer(buffer=img,
                               dtype=image.dtype).reshape(image.shape)
        np.copyto(dst=img_np, src=image)

        # Number of x starting points.
        x_steps = int((image.shape[0] - patch_size) / patch_size *
                      inverse_overlap_factor) + 1
        # Number of y starting points.
        y_steps = int((image.shape[1] - patch_size) / patch_size *
                      inverse_overlap_factor) + 1
        # Step size, same for x and y.
        step_size = int(patch_size / inverse_overlap_factor)

        # Create the queues for passing data back and forth.
        in_queue = Queue()
        out_queue = Queue(maxsize=-1)

        # Create the processes for multiprocessing.
        processes = [
            Process(target=find_patch_mp,
                    args=(functools.partial(
                        find_patch,
                        output_folder=output_folder,
                        image=img_np,
                        by_folder=by_folder,
                        image_loc=image_loc,
                        purple_threshold=purple_threshold,
                        purple_scale_size=purple_scale_size,
                        image_ext=image_ext,
                        type_histopath=type_histopath,
                        patch_size=patch_size), in_queue, out_queue))
            for __ in range(num_workers)
        ]
        for p in processes:
            p.daemon = True
            p.start()

        # Put the (x, y) coordinates in the input queue.
        for xy in itertools.product(range(0, x_steps * step_size, step_size),
                                    range(0, y_steps * step_size, step_size)):
            in_queue.put(obj=xy)

        # Store num_workers None values so the processes exit when not enough jobs left.
        for __ in range(num_workers):
            in_queue.put(obj=None)

        num_patches = sum([out_queue.get() for __ in range(x_steps * y_steps)])

        # Join the processes as they finish.
        for p in processes:
            p.join(timeout=1)

        if by_folder:
            print(f"{image_loc}: num outputted windows: {num_patches}")
        else:
            outputted_patches += num_patches

    if not by_folder:
        print(
            f"finished patches from {input_folder} "
            f"with inverse overlap factor {inverse_overlap_factor:.2f} in {time.time() - start_time:.2f} seconds "
            f"outputting in {output_folder} "
            f"for {outputted_patches} patches")
