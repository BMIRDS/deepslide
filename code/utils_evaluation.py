# DeepSlide
# Jason Wei, Behnaz Abdollahi, Saeed Hassanpour

# Methods for evaluation.


import config
from utils import *

from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
import operator
from sklearn.metrics import cohen_kappa_score, f1_score, accuracy_score


from scipy.misc import imsave
from PIL import Image
Image.MAX_IMAGE_PIXELS=1e10
from random import randint
from scipy.stats import mode
import cv2
import skimage.measure
from skimage.transform import rescale, rotate

###########################################
######### THRESHOLD GRID SEARCH ###########
###########################################

#get a prediction for a single whole slide
def get_prediction(patches_pred_file, conf_thresholds):

	classes = list(conf_thresholds.keys())
	class_to_count = {_class:0 for _class in classes} #predicted class distribution per slide

	#looping through all the lines in the file and adding predictions
	patches_pred_lines = open(patches_pred_file, 'r').readlines()[1:]
	for line in patches_pred_lines:
		line_items = line[:-1].split(',')
		line_class = line_items[2]
		line_conf = float(line_items[3])
		if line_class in classes and line_conf > conf_thresholds[line_class]:
			class_to_count[line_class] += 1

	if sum(class_to_count.values()) > 0:
		class_to_percent = {_class:class_to_count[_class]/sum(class_to_count.values()) for _class in class_to_count}
	else:
		class_to_percent = {_class:0 for _class in class_to_count}
	
	#you can implement your own heuristic other than taking the max
	predicted_class = max(class_to_percent.items(), key=operator.itemgetter(1))[0]

	#diagnostic info that could be helpful
	counts_percent = [str(class_to_percent[_class])[:5] for _class in config.classes]
	counts_num = [str(class_to_count[_class])[:5] for _class in config.classes]
	count_line = ','.join(counts_percent) + ',' + ','.join(counts_num)

	#creating the line for output to csv
	line = patches_pred_file.split('/')[-1].replace("csv", "jpg") + ',' + predicted_class + ',' + count_line

	return line

#output predictions for all the whole slides into a csv
def output_all_predictions(patches_pred_folder, output_folder, conf_thresholds):

	#open a new csv file for each set of confidence thresholds used on each set of wsi
	output_file = ""
	for _class in conf_thresholds:
		output_file += _class + str(conf_thresholds[_class])[1:] + '_'
	output_file = output_file[:-1] + '.csv'
	output_csv_path = join(output_folder, output_file)
	confirm_output_folder(basefolder(output_csv_path))
	writer = open(output_csv_path, 'w')
	writer.write('img,predicted,')
	percent_header = ['percent_'+ _class for _class in config.classes]
	count_header = ['count_' + _class for _class in config.classes]
	line = ','.join(percent_header) + ',' + ','.join(count_header) + '\n'
	writer.write(line)

	csv_paths = get_csv_paths(patches_pred_folder)
	for csv_path in csv_paths:
		writer.write(get_prediction(csv_path, conf_thresholds)+'\n')

	writer.close()

#main function for performing the grid search
#first output predictions for each threshold
def grid_search(threshold_search, pred_folder, inference_folder, labels_csv):

	for threshold in threshold_search:
		conf_thresholds = {_class: threshold for _class in config.classes}
		output_all_predictions(pred_folder, inference_folder, conf_thresholds)

###########################################
######## FINDING BEST THRESHOLDS ##########
###########################################

#get the average class accuracy of predictions
def get_scores(gt_labels, prediction_labels):

	assert len(gt_labels) == len(prediction_labels)

	class_to_gt_count = {_class:0 for _class in config.classes}
	class_to_pred_count = {_class:0 for _class in config.classes}
	gts = []
	preds = []

	for file in sorted(gt_labels.keys()):

		gt_label = gt_labels[file]
		pred_label = prediction_labels[file]
		gts.append(gt_label)
		preds.append(pred_label)

		#predicted correct
		if gt_label == pred_label:
			class_to_pred_count[gt_label] += 1

		#add to total
		class_to_gt_count[gt_label] += 1

	conf_matrix = confusion_matrix(gts, preds)
	class_to_acc = {_class: float(class_to_pred_count[_class]) / class_to_gt_count[_class] for _class in class_to_gt_count}
	avg_class_acc = sum(list(class_to_acc.values())) / len(class_to_acc)

	return avg_class_acc, conf_matrix

#parsing filename
def parse_thresholds(csv_path):
	class_to_threshold = {}
	base = basename(csv_path).replace(".csv", "")
	items = base.split('_')
	for item in items:
		subitems = item.split('.')
		_class = subitems[0]
		threshold = float('0.'+subitems[1])
		class_to_threshold[_class] = threshold
	return class_to_threshold

#printing the best accuracy with thresholds
def get_best_acc(labels_csv, inference_folder):

	gt_labels = create_labels(labels_csv)
	prediction_csv_paths = get_csv_paths(inference_folder)
	best_acc = 0
	best_thresholds = None
	best_csv = None

	for prediction_csv_path in prediction_csv_paths:
		prediction_labels = create_labels(prediction_csv_path)
		avg_class_acc, conf_matrix = get_scores(gt_labels, prediction_labels)
		thresholds = parse_thresholds(prediction_csv_path)
		print('thresholds', thresholds, 'has average class accuracy', str(avg_class_acc)[:5]) 
		if best_acc < avg_class_acc:
			best_acc = avg_class_acc
			best_thresholds = thresholds
			best_csv = prediction_csv_path

	print("view these predictions in", best_csv)


#finding the best threshold
def get_best_thresholds(labels_csv, inference_folder):

	gt_labels = create_labels(labels_csv)
	prediction_csv_paths = get_csv_paths(inference_folder)
	best_acc = 0
	best_thresholds = None

	for prediction_csv_path in prediction_csv_paths:
		prediction_labels = create_labels(prediction_csv_path)
		avg_class_acc, conf_matrix = get_scores(gt_labels, prediction_labels)
		thresholds = parse_thresholds(prediction_csv_path)
		if best_acc < avg_class_acc:
			best_acc = avg_class_acc
			best_thresholds = thresholds

	return best_thresholds

#print accuracy and confusion matrix 
def print_final_test_results(labels_csv, inference_folder):

	gt_labels = create_labels(labels_csv)
	prediction_csv_paths = get_csv_paths(inference_folder)

	for prediction_csv_path in prediction_csv_paths:
		prediction_labels = create_labels(prediction_csv_path)
		avg_class_acc, conf_matrix = get_scores(gt_labels, prediction_labels)
		thresholds = parse_thresholds(prediction_csv_path)
		print('test set has final avg class acc:', str(avg_class_acc)[:5])
		print(conf_matrix)



###########################################
############# VISUALIZATION ###############
###########################################

def color_to_np_color(color):

	colors = {	'white': np.array([255, 255, 255]), 
				'pink': np.array([255, 108, 180]),
				'black': np.array([0, 0, 0]),
				'red': np.array([255, 0, 0]), 
		  		'purple': np.array([225, 225, 0]),
				'yellow': np.array([255, 255, 0]), 
				'orange': np.array([255, 127, 80]),
				'blue': np.array([0, 0, 255]),
				'green': np.array([0, 255, 0])}

	return colors[color]

#overlay the predicted dots on an image
def add_predictions_to_image(xy_to_pred_class, image, prediction_to_color):

	for x, y in xy_to_pred_class.keys():
		prediction, confidence = xy_to_pred_class[x, y]
		x = int(x)
		y = int(y)
		
		image[x+112-11:x+112+11, y+112-11:y+112+11, :] = prediction_to_color[prediction]
		#if confidence > class_to_threshold[prediction]: #confident prediction
			#image[x+112-11:x+112+11, y+112-11:y+112+11, :] = prediction_to_color[prediction]
		#else: #not confident prediction
			#image[x+112-3:x+112+3, y+112-3:y+112+3, :] = prediction_to_color[prediction]

	return image

#get the dictionary of predictions
def get_xy_to_pred_class(window_prediction_folder, img_name):

	xy_to_pred_class = {}

	csv_file = join(window_prediction_folder, img_name.split('.')[0]) + ".csv"
	csv_lines = open(csv_file, 'r').readlines()[1:]
	predictions = [line[:-1].split(',') for line in csv_lines]

	for prediction in predictions: 
		x = prediction[0]
		y = prediction[1]
		pred_class = prediction[2]
		confidence = float(prediction[3])

		#implement thresholding
		xy_to_pred_class[(x, y)] = (pred_class, confidence)

	return(xy_to_pred_class)



#main function for visualization
def visualize(wsi_folder, preds_folder, vis_folder, colors):

	#get list of whole slides
	whole_slides = get_all_image_paths(wsi_folder)
	print(len(whole_slides), "whole slides found from", wsi_folder)

	prediction_to_color = {config.classes[i]:color_to_np_color(config.colors[i]) for i in range(config.num_classes)}

	#for each wsi
	for whole_slide in whole_slides:

		#read in the image
		whole_slide_numpy = cv2.imread(whole_slide); print("visualizing", whole_slide, "of shape", whole_slide_numpy.shape); assert whole_slide_numpy.shape[2] == 3
		
		#get the predictions
		xy_to_pred_class = get_xy_to_pred_class(preds_folder, whole_slide.split('/')[-1])
		
		#add the predictions to image
		whole_slide_with_predictions = add_predictions_to_image(xy_to_pred_class, whole_slide_numpy, prediction_to_color)
		
		#save it
		output_path = join(vis_folder, whole_slide.split('/')[-1].split('.')[0]+'_predictions.jpg')
		confirm_output_folder(basefolder(output_path))
		imsave(output_path, whole_slide_with_predictions)

	print('find the visualizations in', vis_folder)




