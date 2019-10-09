import os
import sys
from os import listdir
from os.path import isfile, join

import numpy as np
from scipy.misc import imsave
from PIL import Image
from random import randint
import time
from scipy.stats import mode
import cv2
import openslide

import skimage.measure
from skimage.transform import rescale, rotate
import time
import argparse

compression_factor = 3
window_size = 10000
Image.MAX_IMAGE_PIXELS = 1e10


def output_jpeg_tiles(image_name, output_path):         # converts svs image with meta data into just the jpeg image

    img = openslide.OpenSlide(image_name)
    width, height = img.level_dimensions[0]

    # img_np_rgb = np.zeros((23000, 30000, 3), dtype=np.uint8)
    increment_x = int(width/window_size) + 1
    increment_y = int(height/window_size) + 1

    print("converting", image_name, "with width", width, "and height", height)

    for incre_x in range(increment_x):         # have to read the image in patches since it doesn't let me do it for larger things
        for incre_y in range(increment_y):

            begin_x = window_size*incre_x
            end_x = min(width, begin_x+window_size)
            begin_y = window_size*incre_y
            end_y = min(height, begin_y+window_size)
            patch_width = end_x-begin_x
            patch_height = end_y-begin_y

            patch = img.read_region((begin_x, begin_y), 0, (patch_width, patch_height))
            patch.load()
            patch_rgb = Image.new("RGB", patch.size, (255, 255, 255))
            patch_rgb.paste(patch, mask=patch.split()[3])

            # compress the image
            patch_rgb = patch_rgb.resize((int(patch_rgb.size[0]/compression_factor), int(patch_rgb.size[1]/compression_factor)), Image.ANTIALIAS)

            # save the image
            output_subfolder = join(output_path, image_name.split('/')[-1][:-4])
            if not os.path.exists(output_subfolder):
                os.makedirs(output_subfolder)
            output_image_name = join(output_subfolder, image_name.split('/')[-1][:-4] + '_' + str(incre_x) + '_' + str(incre_y) + '.jpg')
            # print(output_image_name)
            patch_rgb.save(output_image_name)

            # patch_np = np.swapaxes(np.asarray(patch_rgb), 0, 1)
            # img_np_rgb[begin_x:end_x, begin_y:end_y, :] = patch_np
            # print(sys.getsizeof(img_np_rgb))
            # import psutil
            # process = psutil.Process(os.getpid())
            # print('1', process.memory_info().rss)
            # imsave(output_path[:-5] + '/' + str(incre_x) + str(incre_y) + 'test.jpeg', img_np_rgb)
            # print("bs saved")
            # img_np_rgb[begin_x:end_x, begin_y:end_y, :] = np.swapaxes(np.asarray(patch_rgb), 0, 1)
            # im = Image.fromarray(img_np_rgb)
            # print('2', process.memory_info().rss)
            # print(sys.getsizeof(img_np_rgb))

    # print("about to save image")
    # print(sys.getsizeof(img_np_rgb))
    # imsave(output_path, img_np_rgb)
    # print("image saved")


parser = argparse.ArgumentParser()
parser.add_argument("--input_folder", type=str, help="input folder")
parser.add_argument("--output_folder", type=str, help="output folder")
parser.add_argument("--start_at", type=str, default=None, help="resume from a certain filename")
args = parser.parse_args()


input_folder = args.input_folder
output_folder = args.output_folder
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

image_names = [f for f in listdir(input_folder) if isfile(join(input_folder, f))]
if '.DS_Store' in image_names:
    image_names.remove('.DS_Store')

if args.start_at is not None:
    start = image_names.index(args.start_at)
    print("skipping the first", start)
    image_names = image_names[start+2:]

for image_name in image_names:
    full_image_path = input_folder + '/' + image_name
    output_path = output_folder + '/'
    output_jpeg_tiles(full_image_path, output_path)
