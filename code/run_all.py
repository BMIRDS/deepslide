import config
from utils_evaluation import (find_best_acc_and_thresh, grid_search,
                              output_all_predictions, print_final_test_results,
                              visualize)
from utils_model import (get_predictions, train_resnet)
from utils_processing import (balance_classes, gen_train_patches,
                              gen_val_patches, produce_patches)
from utils_split import split

print("+++++ Running 1_split.py +++++")
split(all_wsi=config.args.all_wsi,
      classes=config.classes,
      keep_orig_copy=config.args.keep_orig_copy,
      labels_test=config.args.labels_test,
      labels_train=config.args.labels_train,
      labels_val=config.args.labels_val,
      test_wsi_per_class=config.args.test_wsi_per_class,
      val_wsi_per_class=config.args.val_wsi_per_class,
      wsi_test=config.args.wsi_test,
      wsi_train=config.args.wsi_train,
      wsi_val=config.args.wsi_val)
print("+++++ Finished running 1_split.py +++++")

###########################################
#                  MAIN                   #
###########################################
print("+++++ Running 2_process_patches.py +++++")
print("----- Generating training patches -----")
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
print("----- Finished generating training patches -----")
print("----- Balancing the training patches -----")
# Balance the training patches.
balance_classes(training_folder=config.train_patches)
print("----- Finished balancing the training patches -----")
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
print("----- Finished generating validation patches -----")
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
print("----- Finished generating validation evaluation patches -----")
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
print("----- Finished generating test evaluation patches -----")
print("+++++ Finished running 2_process_patches.py +++++")

# Training the ResNet.
print("+++++ Running 3_train.py +++++")
train_resnet(batch_size=config.args.batch_size,
             checkpoints_folder=config.args.checkpoints_folder,
             classes=config.classes,
             color_jitter_brightness=config.args.color_jitter_brightness,
             color_jitter_contrast=config.args.color_jitter_contrast,
             color_jitter_hue=config.args.color_jitter_hue,
             color_jitter_saturation=config.args.color_jitter_saturation,
             device=config.device,
             learning_rate=config.args.learning_rate,
             learning_rate_decay=config.args.learning_rate_decay,
             log_csv=config.log_csv,
             num_classes=config.num_classes,
             num_layers=config.args.num_layers,
             num_workers=config.args.num_workers,
             path_mean=config.path_mean,
             path_std=config.path_std,
             pretrain=config.args.pretrain,
             resume_checkpoint=config.args.resume_checkpoint,
             resume_checkpoint_path=config.resume_checkpoint_path,
             save_interval=config.args.save_interval,
             total_epochs=config.args.num_epochs,
             train_folder=config.args.train_folder,
             weight_decay=config.args.weight_decay)
print("+++++ Finished running 3_train.py +++++")

# Run the ResNet on the generated patches.
print("+++++ Running 4_test.py +++++")
print("----- Finding validation patch predictions -----")
# Validation patches.
get_predictions(patches_eval_folder=config.args.patches_eval_val,
                output_folder=config.args.preds_val,
                auto_select=config.args.auto_select,
                batch_size=config.args.batch_size,
                checkpoints_folder=config.args.checkpoints_folder,
                classes=config.classes,
                device=config.device,
                eval_model=config.eval_model,
                num_classes=config.num_classes,
                num_layers=config.args.num_layers,
                num_workers=config.args.num_workers,
                path_mean=config.path_mean,
                path_std=config.path_std,
                pretrain=config.args.pretrain)
print("----- Finished finding validation patch predictions -----")
print("----- Finding test patch predictions -----")
# Test patches.
get_predictions(patches_eval_folder=config.args.patches_eval_test,
                output_folder=config.args.preds_test,
                auto_select=config.args.auto_select,
                batch_size=config.args.batch_size,
                checkpoints_folder=config.args.checkpoints_folder,
                classes=config.classes,
                device=config.device,
                eval_model=config.eval_model,
                num_classes=config.num_classes,
                num_layers=config.args.num_layers,
                num_workers=config.args.num_workers,
                path_mean=config.path_mean,
                path_std=config.path_std,
                pretrain=config.args.pretrain)
print("----- Finished finding test patch predictions -----")
print("+++++ Finished running 4_test.py +++++")

# Searching over thresholds for filtering noise.
print("+++++ Running 5_grid_search.py +++++")
print("----- Running grid search -----")
grid_search(pred_folder=config.args.preds_val,
            inference_folder=config.args.inference_val,
            classes=config.classes,
            image_ext=config.args.image_ext,
            threshold_search=config.threshold_search)
print("----- Finished running grid search -----")
print("+++++ Finished running 5_grid_search.py +++++")

# Visualizing patch predictions with overlaid dots.
print("+++++ Running 6_visualize.py +++++")
print("----- Visualizing validation set -----")
# Visualize validation set.
visualize(wsi_folder=config.args.wsi_val,
          preds_folder=config.args.preds_val,
          vis_folder=config.args.vis_val,
          classes=config.classes,
          colors=config.colors,
          num_classes=config.num_classes,
          patch_size=config.args.patch_size)
print("----- Finished visualizing validation set -----")
print("----- Visualizing test set -----")
# Visualize test set.
visualize(wsi_folder=config.args.wsi_test,
          preds_folder=config.args.preds_test,
          vis_folder=config.args.vis_test,
          classes=config.classes,
          colors=config.colors,
          num_classes=config.num_classes,
          patch_size=config.args.patch_size)
print("----- Finished visualizing test set -----")
print("+++++ Finished running 6_visualize.py +++++")

# Running the code on the testing set.
print("+++++ Running 7_final_test.py +++++")
print("----- Finding best thresholds -----")
best_thresholds = find_best_acc_and_thresh(
    labels_csv=config.args.labels_val,
    inference_folder=config.args.inference_val,
    classes=config.classes)
print("----- Finished finding best thresholds -----")
print("----- Outputting all predictions -----")
output_all_predictions(patches_pred_folder=config.args.preds_test,
                       output_folder=config.args.inference_test,
                       conf_thresholds=best_thresholds,
                       classes=config.classes,
                       image_ext=config.args.image_ext)
print("----- Finished outputting all predictions -----")
print("----- Printing final test results -----")
print_final_test_results(labels_csv=config.args.labels_test,
                         inference_folder=config.args.inference_test,
                         classes=config.classes)
print("----- Finished printing final test results -----")
print("+++++ Finished running 7_final_test.py +++++")
