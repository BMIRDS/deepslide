import config
from utils_evaluation import visualize

# Visualizing patch predictions with overlaid dots.
print("\n\n+++++ Running 6_visualize.py +++++")
print("\n----- Visualizing validation set -----")
# Visualize validation set.
visualize(wsi_folder=config.args.wsi_val,
          preds_folder=config.args.preds_val,
          vis_folder=config.args.vis_val,
          classes=config.classes,
          colors=config.colors,
          num_classes=config.num_classes,
          patch_size=config.args.patch_size)
print("----- Finished visualizing validation set -----\n")
print("----- Visualizing test set -----")
# Visualize test set.
visualize(wsi_folder=config.args.wsi_test,
          preds_folder=config.args.preds_test,
          vis_folder=config.args.vis_test,
          classes=config.classes,
          colors=config.colors,
          num_classes=config.num_classes,
          patch_size=config.args.patch_size)
print("----- Finished visualizing test set -----\n")
print("+++++ Finished running 6_visualize.py +++++\n\n")
