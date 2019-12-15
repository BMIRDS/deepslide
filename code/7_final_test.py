import config
from utils_evaluation import (find_best_acc_and_thresh, output_all_predictions,
                              print_final_test_results)

# Running the code on the testing set.
print("\n\n+++++ Running 7_final_test.py +++++")
print("\n----- Finding best thresholds -----")
best_thresholds = find_best_acc_and_thresh(
    labels_csv=config.args.labels_val,
    inference_folder=config.args.inference_val,
    classes=config.classes)
print("----- Finished finding best thresholds -----\n")
print("----- Outputting all predictions -----")
output_all_predictions(patches_pred_folder=config.args.preds_test,
                       output_folder=config.args.inference_test,
                       conf_thresholds=best_thresholds,
                       classes=config.classes,
                       image_ext=config.args.image_ext)
print("----- Finished outputting all predictions -----\n")
print("----- Printing final test results -----")
print_final_test_results(labels_csv=config.args.labels_test,
                         inference_folder=config.args.inference_test,
                         classes=config.classes)
print("----- Finished printing final test results -----\n")
print("+++++ Finished running 7_final_test.py +++++\n\n")
