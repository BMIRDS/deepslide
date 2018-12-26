# DeepSlide
# Jason Wei, Behnaz Abdollahi, Saeed Hassanpour

#This splits the train, val, and test data.

import config
from utils import *

#main function
#note that we want the validation and test sets to be balanced
def split(all_wsi, train_folder, val_folder, test_folder, val_split, test_split, keep_orig_copy, labels_train, labels_val, labels_test):

	head = 'cp' if keep_orig_copy else 'mv' #based on whether we want to move or keep the files

	#create folders
	for folder in [train_folder, val_folder, test_folder]:
		subfolders = [join(folder, _class) for _class in config.classes]
		for subfolder in subfolders:
			confirm_output_folder(subfolder) 

	train_img_to_label = {}
	val_img_to_label = {}
	test_img_to_label = {}

	#sort the images and move/copy them appropriately
	subfolder_paths = get_subfolder_paths(all_wsi)
	for subfolder in subfolder_paths:

		image_paths = get_image_paths(subfolder)
		assert len(image_paths) > val_split + test_split #make sure we have enough slides in each class

		#assign training, test, and val images
		test_idx = len(image_paths) - test_split
		val_idx = test_idx - val_split
		train_images = image_paths[:val_idx]
		val_images = image_paths[val_idx:test_idx]
		test_images = image_paths[test_idx:]
		print('class '+subfolder.split('/')[-1]+ ':', '#train='+str(len(train_images)), '#val='+str(len(val_images)), '#test='+str(len(test_images)))

		#move train
		for train_image in train_images:
			output_path = '/'.join([train_folder, '/'.join(train_image.split('/')[1:])])
			os.system(' '.join([head, train_image, output_path]))
			img_name = train_image.split('/')[-1]
			img_class = train_image.split('/')[-2]
			train_img_to_label[img_name] = img_class
			#writer_train.write(img_name + ',' + img_class + '\n')

		#move val
		for val_image in val_images:
			output_path = '/'.join([val_folder, '/'.join(val_image.split('/')[1:])])
			os.system(' '.join([head, val_image, output_path]))
			img_name = val_image.split('/')[-1]
			img_class = val_image.split('/')[-2]
			val_img_to_label[img_name] = img_class
			#writer_val.write(img_name + ',' + img_class + '\n')

		#move test
		for test_image in test_images:
			output_path = '/'.join([test_folder, '/'.join(test_image.split('/')[1:])])
			os.system(' '.join([head, test_image, output_path]))
			img_name = test_image.split('/')[-1]
			img_class = test_image.split('/')[-2]
			test_img_to_label[img_name] = img_class
			#writer_test.write(img_name + ',' + img_class + '\n')


	#for making the csv files
	writer_train = open(labels_train, 'w')
	writer_train.write('img,gt\n')
	for img in sorted(train_img_to_label.keys()):
		writer_train.write(img + ',' + train_img_to_label[img] + '\n')

	writer_val = open(labels_val, 'w')
	writer_val.write('img,gt\n')
	for img in sorted(val_img_to_label.keys()):
		writer_val.write(img + ',' + val_img_to_label[img] + '\n')

	writer_test = open(labels_test, 'w')
	writer_test.write('img,gt\n')
	for img in sorted(test_img_to_label.keys()):
		writer_test.write(img + ',' + test_img_to_label[img] + '\n')

if __name__ == '__main__':

	split(	all_wsi = config.all_wsi, 
			train_folder = config.wsi_train, 
			val_folder = config.wsi_val, 
			test_folder = config.wsi_test, 
			val_split = config.val_wsi_per_class, 
			test_split = config.test_wsi_per_class,
			keep_orig_copy = config.keep_orig_copy,
			labels_train = config.labels_train,
			labels_val = config.labels_val,
			labels_test = config.labels_test
		)

