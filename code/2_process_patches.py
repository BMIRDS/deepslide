from utils_processing import (
    gen_train_patches,
    balance_classes,
    gen_val_patches,
    gen_patches_by_folder,
)

###########################################
#                  MAIN                   #
###########################################

# generate train_patches
gen_train_patches(config.wsi_train, config.train_patches,
                  config.num_train_per_class)

# balance the training patches
balance_classes(config.train_patches)

# generate val patches
gen_val_patches(config.wsi_val, config.val_patches, overlap_factor=1.5)

# generate train_eval_patches (this will probably take up a lot of space, so only use for serious debugging)
# gen_patches_by_folder(config.wsi_train, config.patches_eval_train, config.slide_overlap)

# generate val_eval_patches
gen_patches_by_folder(config.wsi_val, config.patches_eval_val,
                      config.slide_overlap)

# generate test_eval_patches
gen_patches_by_folder(config.wsi_test, config.patches_eval_test,
                      config.slide_overlap)
