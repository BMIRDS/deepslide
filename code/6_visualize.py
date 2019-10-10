# DeepSlide
# Jason Wei, Behnaz Abdollahi, Saeed Hassanpour

# Visualizing patch predictions with overlaid dots

from utils_evaluation import visualize

# visualize val set
visualize(wsi_folder=config.wsi_val,
          preds_folder=config.preds_val,
          vis_folder=config.vis_val,
          colors=config.colors)

# visualize test set
# visualize(wsi_folder=config.wsi_test,
#             preds_folder=config.preds_test,
#             vis_folder=config.vis_test,
#             colors=config.colors
#            )
