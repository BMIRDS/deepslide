"""
DeepSlide
Computes the image statistics for normalization.

Authors: Naofumi Tomita
"""

from pathlib import Path
from typing import (List, Tuple)

import torch
from PIL import Image
from torchvision.transforms import ToTensor

Image.MAX_IMAGE_PIXELS = None


def compute_stats(folderpath: Path,
                  image_ext: str) -> Tuple[List[float], List[float]]:
    """
    Compute the mean and standard deviation of the images found in folderpath.

    Args:
        folderpath: Path containing images.
        image_ext: Extension of the image files.

    Returns:
        A tuple containing the mean and standard deviation for the images over the channel, height, and width axes.

    This implementation is based on the discussion from: 
        https://discuss.pytorch.org/t/about-normalization-using-pre-trained-vgg16-networks/23560/9
    """
    class MyDataset(torch.utils.data.Dataset):
        """
        Creates a dataset by reading images.

        Attributes:
            data: List of the string image filenames.
        """
        def __init__(self, folder: Path) -> None:
            """
            Create the MyDataset object.

            Args:
                folder: Path to the images.
            """
            self.data = []

            for file in folder.rglob(f"*{image_ext}"):
                if not file.name.startswith("."):
                    self.data.append(file)

        def __getitem__(self, index: int) -> torch.Tensor:
            """
            Finds the specified image and outputs in correct format.

            Args:
                index: Index of the desired image.

            Returns:
                A PyTorch Tensor in the correct color space.
            """
            return ToTensor()(Image.open(self.data[index]).convert("RGB"))

        def __len__(self) -> int:
            return len(self.data)

    def online_mean_and_sd(loader: torch.utils.data.DataLoader
                           ) -> Tuple[List[float], List[float]]:
        """
        Computes the mean and standard deviation online.
            Var[x] = E[X^2] - (E[X])^2

        Args:
            loader: The PyTorch DataLoader containing the images to iterate over.

        Returns:
            A tuple containing the mean and standard deviation for the images over the channel, height, and width axes.
        """
        cnt = 0
        fst_moment = torch.empty(3)
        snd_moment = torch.empty(3)

        for data in loader:
            b, __, h, w = data.shape
            nb_pixels = b * h * w
            fst_moment = (cnt * fst_moment +
                          torch.sum(data, dim=[0, 2, 3])) / (cnt + nb_pixels)
            snd_moment = (cnt * snd_moment + torch.sum(
                data**2, dim=[0, 2, 3])) / (cnt + nb_pixels)
            cnt += nb_pixels
        return fst_moment.tolist(), torch.sqrt(snd_moment -
                                               fst_moment**2).tolist()

    return online_mean_and_sd(
        loader=torch.utils.data.DataLoader(dataset=MyDataset(
            folder=folderpath),
                                           batch_size=1,
                                           num_workers=1,
                                           shuffle=False))
