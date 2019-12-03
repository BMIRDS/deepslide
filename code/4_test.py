from utils_model import get_predictions
import config

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
