from utils_evaluation import visualize
import config

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
