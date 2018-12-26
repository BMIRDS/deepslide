# DeepSlide
# Jason Wei, Behnaz Abdollahi, Saeed Hassanpour

# Run the resnet on generated patches.

import utils_model
from utils_model import *

#validation patches
get_predictions(	patches_eval_folder = config.patches_eval_val, 
					auto_select = config.auto_select, 
					eval_model = config.eval_model, 
					checkpoints_folder = config.checkpoints_folder, 
					output_folder = config.preds_val)

#test patches
get_predictions(	patches_eval_folder = config.patches_eval_test, 
					auto_select = config.auto_select, 
					eval_model = config.eval_model, 
					checkpoints_folder = config.checkpoints_folder, 
					output_folder = config.preds_test)