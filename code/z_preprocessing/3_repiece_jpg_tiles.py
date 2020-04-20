import argparse
import os
from os import listdir
from os.path import isfile, join, isdir

from PIL import Image

compression_factor = 3
window_size = 10000
Image.MAX_IMAGE_PIXELS = 1e10
compressed_window_size = int(window_size / compression_factor)


def get_image_paths(folder):
    image_paths = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f))]
    if join(folder, '.DS_Store') in image_paths:
        image_paths.remove(join(folder, '.DS_Store'))
    return image_paths


def get_subfolder_paths(folder):
    subfolder_paths = [join(folder, f) for f in listdir(folder) if (isdir(join(folder, f)) and '.DS_Store' not in f)]
    if join(folder, '.DS_Store') in subfolder_paths:
        subfolder_paths.remove(join(folder, '.DS_Store'))
    return subfolder_paths


def get_num_horizontal_positions(input_folder):
    horizontal_positions = []
    image_paths = get_image_paths(input_folder)
    for image_path in image_paths:
        x_increment = int(image_path.split('/')[-1].split('.')[0].split('_')[1])
        horizontal_positions.append(x_increment)
    return len(set(horizontal_positions))


def get_num_vertical_positions(input_folder):
    vertical_positions = []
    image_paths = get_image_paths(input_folder)
    for image_path in image_paths:
        x_increment = int(image_path.split('/')[-1].split('.')[0].split('_')[2])
        vertical_positions.append(x_increment)
    return len(set(vertical_positions))


def output_repieced_image(input_folder, output_image_path):

    num_horizontal_positions = get_num_horizontal_positions(input_folder)
    num_vertical_positions = get_num_vertical_positions(input_folder)

    image_paths = get_image_paths(input_folder)
    images = map(Image.open, image_paths)
    widths, heights = zip(*(i.size for i in images))

    last_width = min(widths)
    last_height = min(heights)

    total_width = (num_horizontal_positions - 1)*compressed_window_size + last_width
    total_height = (num_vertical_positions - 1)*compressed_window_size + last_height

    new_im = Image.new('RGB', (total_width, total_height))

    for image_path in image_paths:

        x_increment = int(image_path.split('/')[-1].split('.')[0].split('_')[1])
        y_increment = int(image_path.split('/')[-1].split('.')[0].split('_')[2])

        image = Image.open(image_path)
        new_im.paste(image, (compressed_window_size*x_increment, compressed_window_size*y_increment))

    new_im.save(output_image_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, help="input folder")
    parser.add_argument("--output_folder", type=str, help="output folder")
    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    input_subfolders = get_subfolder_paths(input_folder)
    for input_subfolder in input_subfolders:
        output_image_path = join(output_folder, input_subfolder.split('/')[-1]+'.jpg')
        print(input_subfolder, output_image_path)
        output_repieced_image(input_subfolder, output_image_path)
