import os
from os import listdir
from os.path import isfile, join
import argparse


def get_image_paths(folder):
    image_paths = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f))]
    if join(folder, '.DS_Store') in image_paths:
        image_paths.remove(join(folder, '.DS_Store'))
    return image_paths


parser = argparse.ArgumentParser()
parser.add_argument("--input_folder", type=str, help="input folder")
args = parser.parse_args()

image_paths = get_image_paths(args.input_folder)
for image_path in image_paths:
    clean_path = image_path.replace(' ', '')
    clean_path = image_path.replace('_', '-')
    print(clean_path)
    os.rename(image_path, clean_path)
print(image_paths)
