from utils_evaluation import (find_best_acc_and_thresh, output_all_predictions,
                              print_final_test_results)
import config

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
