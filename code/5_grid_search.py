# DeepSlide
# Jason Wei, Behnaz Abdollahi, Saeed Hassanpour

# Searching over thresholds for filtering noise

import utils_evaluation
from utils_evaluation import *

grid_search(threshold_search = config.threshold_search,
			pred_folder = config.preds_val, 
			inference_folder = config.inference_val, 
			labels_csv = config.labels_val)

get_best_acc(	labels_csv = config.labels_val, 
				inference_folder = config.inference_val)