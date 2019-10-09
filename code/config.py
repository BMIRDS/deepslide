# DeepSlide
# Jason Wei, Behnaz Abdollahi, Saeed Hassanpour

# This file contains all hyperparameters for the entire repository.

import utils
from utils import *

###########################################
#               USER INPUTS               #
###########################################

# input folder for training images
# must contain subfolders of images labeled by class
# if your two classes are 'a' and 'n', you must have a/*.jpg with the images in class a and n/*.jpg with images in class n
all_wsi = 'all_wsi'
val_wsi_per_class = 20 			# for splitting into validation set
test_wsi_per_class = 30 		# for splitting into testing set, remaining images used in train
keep_orig_copy = True 			# when splitting, do you want to just move them or make a copy?

###########################################
#                GENERAL                  #
###########################################

classes = get_classes(all_wsi) 		# automatically read in the classes
num_classes = len(classes)
patch_size = 224 					# default shape for resnet in pytorch. i would think hard before changing this

###########################################
#               DATA SPLIT                #
###########################################

# the names of your to-be folders
wsi_train = 'wsi_train'
wsi_val = 'wsi_val'
wsi_test = 'wsi_test'

# where the csv file labels will go
labels_train = 'labels_train.csv'
labels_val = 'labels_val.csv'
labels_test = 'labels_test.csv'

###########################################
#     PROCESSING AND PATCH GENERATION     #
###########################################

# this is the input for model training, automatically built
train_folder = 'train_folder'
train_patches = 'train_folder/train'
val_patches = 'train_folder/val'

# folders of patches by wsi in training set, used for getting train acc on wsi leve;
patches_eval_train = 'patches_eval_train'

# folders of patches by wsi in val set, used for getting val acc on wsi level
patches_eval_val = 'patches_eval_val'

# folders of patches by wsi in test set, used for getting test acc on wsi level
patches_eval_test = 'patches_eval_test'

# target number of training samples per class
num_train_per_class = 80000

# only looks for purple images and automatically filters whitespace
type_histopath = True

# sliding window overlap factor (for testing)
# for generating patches during the testing phase, we slide a window to overlap by some factor
# must be an integer. 1 means no overlap, 2 means overlap by 1/2, 3 means overlap by 1/3.
# recommend 2 for very high res, 3 for medium, and 5 for not extremely high res images
slide_overlap = 3

###########################################
#                TRAINING                 #
###########################################

# model hyperparamters
num_epochs = 20
num_layers = 18 									# choose from [18, 32, 50, 101, 152]
learning_rate = 0.001
batch_size = 16
weight_decay = 1e-4
learning_rate_decay = 0.85
resume_checkpoint = False
resume_checkpoint_path = 'checkpoints/xyz.pt'	 	# only used if resume_checkpoint is True
save_interval = 1
checkpoints_folder = 'checkpoints'		 			# where models are saved
pretrain = False 									# imagenet pretrain?
log_folder = 'logs'
log_csv = get_log_csv_name(log_folder) 				# is named with date and time

###########################################
#               PREDICTION                #
###########################################

# selecting the best model
auto_select = True 						# automatically selects the model with the highest validation accuracy
eval_model = 'checkpoints/xyz.pt' 		# does nothing if auto_select is true

preds_train = 'preds_train' 			# where to put the training prediction csv files
preds_val = 'preds_val' 				# where to put the validation prediction csv files
preds_test = 'preds_test' 				# where to put the testing prediction csv files

###########################################
#               EVALUATION                #
###########################################

# find the best threshold for filtering noise (discard patches with a confidence less than this threshold)
threshold_search = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# folder for outputing wsi predictions based on each threshold
inference_train = 'inference_train'
inference_val = 'inference_val'
inference_test = 'inference_test'

# for visualization
colors = ['red', 'white', 'blue', 'green', 'purple', 'orange', 'black', 'pink', 'yellow'] 	# this order is the same order as your sorted classes
vis_train = 'vis_train'
vis_val = 'vis_val'
vis_test = 'vis_test'
