#Author: Jason Wei
#Date: 04/21/2018
#Email: jason.20@dartmouth.edu

#Fetch all the arguments from the command line
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--input_folder", type=str, help="input folder")
parser.add_argument("--compression_factor", type=float, default=5)
parser.add_argument("--output_folder", type=str, help="output_folder")
args = parser.parse_args()
input_folder = args.input_folder
compression_factor = args.compression_factor
assert input_folder is not None

#Get the paths to the images
import os
from os import listdir
from os.path import isfile, join
image_names = [f for f in listdir(input_folder) if isfile(join(input_folder, f))]
if '.DS_Store' in image_names:
	image_names.remove('.DS_Store')
image_names = sorted(image_names)
image_names = image_names[60:]
print(len(image_names), "images found")

############################################
############################################
########## actual algorithm part ###########
############################################
############################################

import numpy as np
from scipy.misc import imsave
from PIL import Image
Image.MAX_IMAGE_PIXELS=1e10
from random import randint
import time
from scipy.stats import mode
import cv2

import skimage.measure
from skimage.transform import rescale, rotate


output_folder = args.output_folder#input_folder+'_'+str(compression_factor)

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

#Get all the crops
start_time = time.time()

for i, image_name in enumerate(image_names):
	image_path = join(input_folder, image_name)
	
	print("loading", image_path)
	image = cv2.imread(image_path)
	print("loaded image from", image_path, "with shape", image.shape, "compressing by", compression_factor)

	if not compression_factor == 1:
		image = rescale(image, 1/compression_factor)
		image = np.rint(image*256)

	imsave(join(output_folder, image_name), image)
	print(i, "/", len(image_names), "saved")

print("code finished")
total_time = time.time() - start_time
print('total time : ', total_time)
print('processing time per image', total_time / len(image_names))



















