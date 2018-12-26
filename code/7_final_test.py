# DeepSlide
# Jason Wei, Behnaz Abdollahi, Saeed Hassanpour

# Running the code on the testing set

import utils_evaluation
from utils_evaluation import *

best_thresholds = get_best_thresholds(	labels_csv = config.labels_val, 
								inference_folder = config.inference_val)

output_all_predictions(	patches_pred_folder = config.preds_test,
						output_folder = config.inference_test,
						conf_thresholds = best_thresholds)

print_final_test_results(labels_csv = config.labels_test, 
						inference_folder = config.inference_test)