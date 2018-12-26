# DeepSlide
# Jason Wei, Behnaz Abdollahi, Saeed Hassanpour

# General helper methods used in other functions.

import math
import time
import numpy as np
from random import randint
import datetime

import os
from os import listdir
from os.path import isfile, join, isdir

#getting the classes for classification
def get_classes(folder):
	subfolder_paths = sorted([f for f in listdir(folder) if (isdir(join(folder, f)) and '.DS_Store' not in f)])
	return subfolder_paths

#log csv name
def get_log_csv_name(log_folder):
	now = datetime.datetime.now()
	month, day, year, hour, minute, second = now.month, now.day, now.year, now.hour, now.minute, now.second
	return log_folder + '/log_' + str(month) + str(day) + str(year) + '_' + str(hour) + str(minute) + str(second) + '.csv'

#just get the name of images in a folder
def get_image_names(folder):
	image_names = [f for f in listdir(folder) if isfile(join(folder, f))]
	if '.DS_Store' in image_names:
		image_names.remove('.DS_Store')
	image_names = sorted(image_names)
	return image_names

#get full image paths
def get_image_paths(folder):
	image_paths = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f))]
	if join(folder, '.DS_Store') in image_paths:
		image_paths.remove(join(folder, '.DS_Store'))
	image_paths = sorted(image_paths)
	return image_paths

#get subfolders
def get_subfolder_paths(folder):
	subfolder_paths = [join(folder, f) for f in listdir(folder) if (isdir(join(folder, f)) and '.DS_Store' not in f)]
	if join(folder, '.DS_Store') in subfolder_paths:
		subfolder_paths.remove(join(folder, '.DS_Store'))
	subfolder_paths = sorted(subfolder_paths)
	return subfolder_paths

#create an output folder if it does not already exist
def confirm_output_folder(output_folder):
	if not os.path.exists(output_folder):
	    os.makedirs(output_folder)

#get 'x' from 'x.jpg'
def file_no_extension(file):
	head = file.split('.')[:-1]
	return "".join(head)

#get '17asdfasdf2d_0_0.jpg' from 'train_folder/train/o/17asdfasdf2d_0_0.jpg'
def basename(path):
	return path.split('/')[-1]

#get 'train_folder/train/o' from 'train_folder/train/o/17asdfasdf2d_0_0.jpg'
def basefolder(path):
	return '/'.join(path.split('/')[:-1])

#get all image paths
def get_all_image_paths(master_folder):

	all_paths = []
	subfolders = get_subfolder_paths(master_folder)
	if len(subfolders) > 1:
		for subfolder in subfolders:
			all_paths += get_image_paths(subfolder)
	else:
		all_paths = get_image_paths(master_folder)
	return all_paths

#get csv paths in a folder
def get_csv_paths(folder):
	csv_paths = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f)) and 'csv' in f]
	if join(folder, '.DS_Store') in csv_paths:
		csv_paths.remove(join(folder, '.DS_Store'))
	csv_paths = list(sorted(csv_paths))
	return csv_paths

#reading labels from a csv
def create_labels(csv_path):

	lines = open(csv_path, 'r').readlines()[1:]
	file_to_gt_label = {}

	for line in lines:
		if len(line) > 3:
			pieces = line[:-1].split(',')
			file = pieces[0]
			gt_label = pieces[1]
			file_to_gt_label[file] = gt_label

	return file_to_gt_label



