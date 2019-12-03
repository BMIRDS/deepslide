import config
from utils_evaluation import (find_best_acc_and_thresh, grid_search,
                              output_all_predictions, print_final_test_results,
                              visualize)
from utils_model import (get_predictions, train_resnet)
from utils_processing import (balance_classes, gen_train_patches,
                              gen_val_patches, produce_patches)
from utils_split import split

print("+++++ Running 1_split.py +++++")
split()
print("+++++ Finished running 1_split.py +++++")

###########################################
#                  MAIN                   #
###########################################
print("+++++ Running 2_process_patches.py +++++")
print("----- Generating training patches -----")
# Generate training patches.
gen_train_patches(config.args.wsi_train, config.train_patches,
                  config.args.num_train_per_class)
print("----- Finished generating training patches -----")
print("----- Balancing the training patches -----")
# Balance the training patches.
balance_classes(config.train_patches)
print("----- Finished balancing the training patches -----")
print("----- Generating validation patches -----")
# Generate validation patches.
gen_val_patches(config.args.wsi_val, config.val_patches, overlap_factor=1.5)
print("----- Finished generating validation patches -----")
print("----- Generating validation evaluation patches -----")
# Generate validation evaluation patches.
produce_patches(input_folder=config.args.wsi_val,
                output_folder=config.args.patches_eval_val,
                inverse_overlap_factor=config.args.slide_overlap,
                by_folder=True)
print("----- Finished generating validation evaluation patches -----")
print("----- Generating test evaluation patches -----")
# Generate test evaluation patches.
produce_patches(input_folder=config.args.wsi_test,
                output_folder=config.args.patches_eval_test,
                inverse_overlap_factor=config.args.slide_overlap,
                by_folder=True)
print("----- Finished generating test evaluation patches -----")
print("+++++ Finished running 2_process_patches.py +++++")

# Training the ResNet.
print("+++++ Running 3_train.py +++++")
train_resnet()
print("+++++ Finished running 3_train.py +++++")

# Run the ResNet on the generated patches.
print("+++++ Running 4_test.py +++++")
print("----- Finding validation patch predictions -----")
# Validation patches.
get_predictions(patches_eval_folder=config.args.patches_eval_val,
                output_folder=config.args.preds_val)
print("----- Finished finding validation patch predictions -----")
print("----- Finding test patch predictions -----")
# Test patches.
get_predictions(patches_eval_folder=config.args.patches_eval_test,
                output_folder=config.args.preds_test)
print("----- Finished finding test patch predictions -----")
print("+++++ Finished running 4_test.py +++++")

# Searching over thresholds for filtering noise.
print("+++++ Running 5_grid_search.py +++++")
print("----- Running grid search -----")
grid_search(pred_folder=config.args.preds_val,
            inference_folder=config.args.inference_val)
print("----- Finished running grid search -----")
print("+++++ Finished running 5_grid_search.py +++++")

# Visualizing patch predictions with overlaid dots.
print("+++++ Running 6_visualize.py +++++")
print("----- Visualizing validation set -----")
# Visualize validation set.
visualize(wsi_folder=config.args.wsi_val,
          preds_folder=config.args.preds_val,
          vis_folder=config.args.vis_val)
print("----- Finished visualizing validation set -----")
print("----- Visualizing test set -----")
# Visualize test set.
visualize(wsi_folder=config.args.wsi_test,
          preds_folder=config.args.preds_test,
          vis_folder=config.args.vis_test)
print("----- Finished visualizing test set -----")
print("+++++ Finished running 6_visualize.py +++++")

# Running the code on the testing set.
print("+++++ Running 7_final_test.py +++++")
print("----- Finding best thresholds -----")
best_thresholds = find_best_acc_and_thresh(
    labels_csv=config.args.labels_val,
    inference_folder=config.args.inference_val)
print("----- Finished finding best thresholds -----")
print("----- Outputting all predictions -----")
output_all_predictions(patches_pred_folder=config.args.preds_test,
                       output_folder=config.args.inference_test,
                       conf_thresholds=best_thresholds)
print("----- Finished outputting all predictions -----")
print("----- Printing final test results -----")
print_final_test_results(labels_csv=config.args.labels_test,
                         inference_folder=config.args.inference_test)
print("----- Finished printing final test results -----")
print("+++++ Finished running 7_final_test.py +++++")
