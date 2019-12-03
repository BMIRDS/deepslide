import config

from utils_processing import (balance_classes, gen_train_patches,
                              gen_val_patches, produce_patches)

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
