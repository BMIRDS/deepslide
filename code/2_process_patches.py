import config

from utils_processing import (balance_classes, gen_train_patches,
                              gen_val_patches, produce_patches)

###########################################
#                  MAIN                   #
###########################################
print("\n\n+++++ Running 2_process_patches.py +++++")
print("\n----- Generating training patches -----")
# Generate training patches.
gen_train_patches(input_folder=config.args.wsi_train,
                  output_folder=config.train_patches,
                  num_train_per_class=config.args.num_train_per_class,
                  num_workers=config.args.num_workers,
                  patch_size=config.args.patch_size,
                  purple_threshold=config.args.purple_threshold,
                  purple_scale_size=config.args.purple_scale_size,
                  image_ext=config.args.image_ext,
                  type_histopath=config.args.type_histopath)
print("----- Finished generating training patches -----\n")
print("----- Balancing the training patches -----")
# Balance the training patches.
balance_classes(training_folder=config.train_patches)
print("----- Finished balancing the training patches -----\n")
print("----- Generating validation patches -----")
# Generate validation patches.
gen_val_patches(input_folder=config.args.wsi_val,
                output_folder=config.val_patches,
                overlap_factor=config.args.gen_val_patches_overlap_factor,
                num_workers=config.args.num_workers,
                patch_size=config.args.patch_size,
                purple_threshold=config.args.purple_threshold,
                purple_scale_size=config.args.purple_scale_size,
                image_ext=config.args.image_ext,
                type_histopath=config.args.type_histopath)
print("----- Finished generating validation patches -----\n")
print("----- Generating validation evaluation patches -----")
# Generate validation evaluation patches.
produce_patches(input_folder=config.args.wsi_val,
                output_folder=config.args.patches_eval_val,
                inverse_overlap_factor=config.args.slide_overlap,
                by_folder=config.args.by_folder,
                num_workers=config.args.num_workers,
                patch_size=config.args.patch_size,
                purple_threshold=config.args.purple_threshold,
                purple_scale_size=config.args.purple_scale_size,
                image_ext=config.args.image_ext,
                type_histopath=config.args.type_histopath)
print("----- Finished generating validation evaluation patches -----\n")
print("----- Generating test evaluation patches -----")
# Generate test evaluation patches.
produce_patches(input_folder=config.args.wsi_test,
                output_folder=config.args.patches_eval_test,
                inverse_overlap_factor=config.args.slide_overlap,
                by_folder=config.args.by_folder,
                num_workers=config.args.num_workers,
                patch_size=config.args.patch_size,
                purple_threshold=config.args.purple_threshold,
                purple_scale_size=config.args.purple_scale_size,
                image_ext=config.args.image_ext,
                type_histopath=config.args.type_histopath)
print("----- Finished generating test evaluation patches -----\n")
print("+++++ Finished running 2_process_patches.py +++++\n\n")
