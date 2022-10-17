"""
DeepSlide
Computes the image statistics for normalization.

Authors: Naofumi Tomita
"""
import argparse
import json
from datetime import datetime
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

    def online_mean_and_sd(
        loader: torch.utils.data.DataLoader, report_interval: int=1000
                           ) -> Tuple[List[float], List[float]]:
        """
        Computes the mean and standard deviation online.
            Var[x] = E[X^2] - (E[X])^2

        Args:
            loader: The PyTorch DataLoader containing the images to iterate over.
            report_interval: Report the intermediate results every N items. (N=0 to suppress reporting.)

        Returns:
            A tuple containing the mean and standard deviation for the images
            over the channel, height, and width axes.
        """
        cnt = 0
        fst_moment = torch.empty(3)
        snd_moment = torch.empty(3)

        for i, data in enumerate(loader, 1):
            b, __, h, w = data.shape
            nb_pixels = b * h * w
            fst_moment = (cnt * fst_moment +
                          torch.sum(data, dim=[0, 2, 3])) / (cnt + nb_pixels)
            snd_moment = (cnt * snd_moment + torch.sum(
                data**2, dim=[0, 2, 3])) / (cnt + nb_pixels)
            cnt += nb_pixels
            if report_interval != 0 and i % report_interval == 0:
                temp_mean = fst_moment.tolist()
                temp_std = torch.sqrt(snd_moment - fst_moment**2).tolist()
                print(f"Mean: {temp_mean}; STD: {temp_std} at iter: {i}")
        return fst_moment.tolist(), torch.sqrt(snd_moment -
                                               fst_moment**2).tolist()

    return online_mean_and_sd(
        loader=torch.utils.data.DataLoader(
            dataset=MyDataset(folder=folderpath),
            batch_size=1,
            num_workers=1,
            shuffle=False))

def save_stats(mean: List, std: List, datapath: str):
    data = {
        'mean': mean,
        'std': std,
        'datapath': datapath}
    data = json.dumps(data, indent=4)
    filename = f"stats_{datetime.now().strftime('%Y-%m-%d_%H:%M')}.json"
    with open(filename, 'w') as outfile:
        outfile.write(data)

    print(f"Results are saved in {filename}.")

def load_stats(jsonfile: str):
    """ Load a stats file in json and return mean and std in lists.
    """
    with open(jsonfile, 'r') as infile:        
        data = json.load(infile)

    print(f"Stats of \'{data['datapath']}\' are loaded from {jsonfile}.")
    return data['mean'], data['std']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compute channel-wise patch color mean and std.')
    parser.add_argument('--datapath', '-i', type=str, required=True,
        help='Path containing images.')
    parser.add_argument('--image_ext', '-x', type=str, default='.png',
        help='Specify file extension of images. Default: .png')
    parser.add_argument('--report_interval', '-n', type=int, default=1000,
        help='Report the intermediate results every N items. Default: 1000')
    parser.add_argument('--save_results', '-d', action='store_true', default=False,
        help='Set this flag to save results.')
    args = parser.parse_args()

    mean, std = compute_stats(Path(args.datapath), args.image_ext,)
    print(f"Mean: {mean}; STD: {std}")

    if args.save_results:
        save_stats(mean=mean, std=std, datapath=args.datapath)


