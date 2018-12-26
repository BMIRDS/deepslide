# DeepSlide
# Jason Wei, Behnaz Abdollahi, Saeed Hassanpour

# This file contains all the methods for processing

import config
from utils import *

from PIL import Image
Image.MAX_IMAGE_PIXELS=1e10
import cv2

import skimage.measure
from skimage.transform import rescale, rotate
from scipy.stats import mode
from scipy.misc import imsave

###########################################
########### FILTERING WHITESPACE ##########
###########################################

def is_purple_dot(r, g, b):
	rb_avg = (r+b)/2
	if r > g - 10 and b > g - 10 and rb_avg > g + 20:
		return True
	return False
	
#this is actually a better method than is whitespace, but only if your images are purple lols
def is_purple(crop):
	pooled = skimage.measure.block_reduce(crop, (int(crop.shape[0]/15), int(crop.shape[1]/15), 1), np.average)
	num_purple_squares = 0
	for x in range(pooled.shape[0]):
		for y in range(pooled.shape[1]):
			r = pooled[x, y, 0]
			g = pooled[x, y, 1]
			b = pooled[x, y, 2]
			if is_purple_dot(r, g, b):
				num_purple_squares += 1
	if num_purple_squares > 100: 
		return True
	return False



###########################################
######### GENERATING TRAINING DATA ########
###########################################

#returns number of folders and images given a folder path
#used for deciding how much to slide windows
def get_folder_size_and_num_images(folder):
	image_paths = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f))]
	if join(folder, '.DS_Store') in image_paths:
		image_paths.remove(join(folder, '.DS_Store'))
	file_size = 0
	for image_path in image_paths:
		file_size += os.path.getsize(image_path)
		#print(image_path, os.path.getsize(image_path))
	file_size_mb = file_size / 1000.0 / 1000.0
	return file_size_mb, len(image_paths)

#how much should the inverse overlap factor be for each folder so that the class distributions are equal?
#returns a dictionary
def get_subfolder_to_overlap(subfolders, desired_crops_per_class):
	subfolder_to_overlap_factor = {}
	for subfolder in subfolders:
		subfolder_size, subfolder_num_images = get_folder_size_and_num_images(subfolder)	
		overlap_factor = math.sqrt(desired_crops_per_class / (subfolder_size / 0.013)) #each image is 13 kb = 0.013 MB, idk I just added two randomly
		overlap_factor = max(1, math.pow(overlap_factor, 1.5)) #empircal guessing?
		subfolder_to_overlap_factor[subfolder] = overlap_factor
		print(subfolder + ": " + str(subfolder_size)[:9] + "MB, " + str(subfolder_num_images) + " images, overlap_factor=" + str(overlap_factor))
	return subfolder_to_overlap_factor

#zero padding for really small crops
def zero_pad(image, patch_size):

	x = image.shape[0] #get current x and y of image
	y = image.shape[1]
	if x >= patch_size and y >= patch_size:
		return image #if its already big enough, then do nothing

	x_new = max(x, patch_size)
	y_new = max(y, patch_size)
	new_image = np.zeros((x_new, y_new, 3)) #otherwise, make a new image
	x_start = int(x_new/2 - x/2)
	y_start = int(y_new/2 - y/2) #find where to place the old image
	new_image[x_start:x_start+x, y_start:y_start+y, :] = image #place the old image

	return new_image #return the padded image

#get the small windows a given subfolder
#this is a big boy function
def produce_patches(input_subfolder, output_subfolder, inverse_overlap_factor):

	confirm_output_folder(output_subfolder)#make the directory if it doens't exist
	image_names = get_image_names(input_subfolder)
	start_time = time.time()
	outputed_windows_per_subfolder = 0

	print('\n' + "getting small crops from " + str(len(image_names)) + ' images in ' + input_subfolder + " with inverse overlap factor " + str(inverse_overlap_factor) + " outputting in " + output_subfolder)

	#get the patches for each wsi
	for image_name in image_names:

		image_path = join(input_subfolder, image_name)
		image = cv2.imread(image_path)
		image = zero_pad(image, config.patch_size) #zero pad if too small

		x_max = image.shape[0] #width of image
		y_max = image.shape[1] #height of image
		window_size = 224
		x_steps = int((x_max-window_size) / window_size * inverse_overlap_factor) #number of x starting points
		y_steps = int((y_max-window_size) / window_size * inverse_overlap_factor) #number of y starting points
		step_size = int(config.patch_size / inverse_overlap_factor) #step size, same for x and y

		#loop through the entire big image
		for i in range(x_steps+1):
			for j in range(y_steps+1):

				#get a patch
				x_start = i*step_size
				x_end = x_start + config.patch_size
				y_start = j*step_size
				y_end = y_start + config.patch_size
				assert x_start >= 0; assert y_start >= 0; assert x_end <= x_max; assert y_end <= y_max
				patch = image[x_start:x_end, y_start:y_end, :]
				assert patch.shape == (config.patch_size, config.patch_size, 3)
				out_path = join(output_subfolder, file_no_extension(image_name)+"_"+str(x_start)+"_"+str(y_start)+".jpg")

				if config.type_histopath: #do you want to check for white space?
					if is_purple(patch): #if its purple (histopathology images)
						imsave(out_path, patch)
						outputed_windows_per_subfolder += 1

				else:
					imsave(out_path, patch)
					outputed_windows_per_subfolder += 1

	total_time = time.time() - start_time
	print("finished patches from " + input_subfolder + " with inverse overlap factor " + str(inverse_overlap_factor) + " outputting in " + output_subfolder)
	print('total time : ', total_time, 'for', outputed_windows_per_subfolder, 'patches')

#use this function to generate all patches for subfolders in the training set
def gen_train_patches(input_folder, output_folder, num_train_per_class):

	#get the subfolders and how much patches should overlap for each
	subfolders = get_subfolder_paths(input_folder)
	print(len(subfolders), "subfolders found from", input_folder)
	subfolder_to_overlap_factor = get_subfolder_to_overlap(subfolders, num_train_per_class)
	#print(subfolder_to_overlap_factor)

	#produce the patches
	for input_subfolder in subfolders:
		overlap_factor = subfolder_to_overlap_factor[input_subfolder]
		output_subfolder = join(output_folder, input_subfolder.split('/')[-1])
		produce_patches(input_subfolder, output_subfolder, overlap_factor)

	print("\nfinished all folders\n")

#use this function to generate all patches for subfolders in the validation set
def gen_val_patches(input_folder, output_folder, overlap_factor):

	#get the subfolders and how much patches should overlap for each
	subfolders = get_subfolder_paths(input_folder)
	print(len(subfolders), "subfolders found from", input_folder)

	#produce the patches
	for input_subfolder in subfolders:
		output_subfolder = join(output_folder, input_subfolder.split('/')[-1])
		produce_patches(input_subfolder, output_subfolder, overlap_factor)

	print("\nfinished all folders\n")


###########################################
####### BALANCING CLASS DISTRIBUTION ######
###########################################

def duplicate_until_n(image_paths, n):

	num_dupls = n - len(image_paths)

	print('balancing', basefolder(image_paths[0]), 'by duplicating', num_dupls)

	for i in range(num_dupls):
		this_round = int(i / len(image_paths)) + 2
		idx = i % len(image_paths)
		image_path = image_paths[idx]
		dupl_path = basefolder(image_path) + '/' + '_'.join(basename(image_path).split('_')[:-2]) + 'dup' + str(this_round) + '_' + '_'.join(basename(image_path).split('_')[-2:])
		os.system(" ".join(['cp', image_path, dupl_path]))

#balancing class distribution so that training isn't skewed
def balance_classes(training_folder):

	subfolders = get_subfolder_paths(training_folder)
	subfolder_to_images = {subfolder:get_image_paths(subfolder) for subfolder in subfolders}
	subfolder_to_num_images = {subfolder:len(subfolder_to_images[subfolder]) for subfolder in subfolders}

	#get class with the most images
	biggest_size = max(subfolder_to_num_images.values())

	for subfolder in subfolder_to_images:
		image_path = subfolder_to_images[subfolder]
		duplicate_until_n(image_path, biggest_size)

	print('balanced all training classes to have', biggest_size, 'images\n')

###########################################
####### GENERATING PATCHES BY FOLDER ######
###########################################

def add_zeros(string):
	while len(string) < 5:
		string = "0" + string
	return string

#big boy function
def gen_patches_by_folder(input_folder, output_folder, inverse_overlap_factor):
	
	start_time = time.time()
	print('\n' + "getting small crops from " + input_folder + " with inverse overlap factor " + str(inverse_overlap_factor) + " outputting in " + output_folder)
	confirm_output_folder(output_folder)
	image_paths = get_all_image_paths(input_folder)

	#for each wsi...
	for image_path in image_paths:

		#load the image
		image = cv2.imread(image_path)
		x_max = image.shape[0] #width of image
		y_max = image.shape[1] #height of image

		if x_max < 224 or y_max < 224:
			print(image_path, 'of size', x_max, 'by', y_max, "is too small")

		else:
			num_outputed_windows = 0
			x_steps = int((x_max-config.patch_size) / config.patch_size * inverse_overlap_factor) #number of x starting points
			y_steps = int((y_max-config.patch_size) / config.patch_size * inverse_overlap_factor) #number of y starting points
			step_size = int(config.patch_size / inverse_overlap_factor) #step size, same for x and y

			#this is hacky due to the way patches are loaded into pytorch
			output_subsubfolder = join(output_folder, basename(image_path).split('.')[0])
			output_subsubfolder = join(output_subsubfolder, output_subsubfolder.split('/')[-1])
			confirm_output_folder(output_subsubfolder)

			#slide the window
			for i in range(x_steps+1):
				for j in range(y_steps+1):

					x_start = i*step_size
					x_end = x_start + config.patch_size
					y_start = j*step_size
					y_end = y_start + config.patch_size
					assert x_start >= 0; assert y_start >= 0; assert x_end <= x_max; assert y_end <= y_max

					patch = image[x_start:x_end, y_start:y_end, :]
					assert patch.shape == (config.patch_size, config.patch_size, 3)
					out_path = join(output_subsubfolder, add_zeros(str(x_start))+";"+add_zeros(str(y_start))+".jpg")

					if config.type_histopath: #do you want to check for white space?
						if is_purple(patch): #if its purple (histopathology images)
							imsave(out_path, patch)
							num_outputed_windows += 1
					else:
						imsave(out_path, patch)
						num_outputed_windows += 1

			print(image_path, ": num outputed windows:", num_outputed_windows)#, "; percent whitespace:", str(whitespace_ratio)[:6])
			

	total_time = time.time() - start_time
	print("finished generating patches from " + input_folder + " in " + str(total_time) + " seconds " + " outputting in " + output_folder)













