import config
from utils_evaluation import grid_search

# Searching over thresholds for filtering noise.
print("\n\n+++++ Running 5_grid_search.py +++++")
print("\n----- Running grid search -----")
grid_search(pred_folder=config.args.preds_val,
            inference_folder=config.args.inference_val,
            classes=config.classes,
            image_ext=config.args.image_ext,
            threshold_search=config.threshold_search)
print("----- Finished running grid search -----\n")
print("+++++ Finished running 5_grid_search.py +++++\n\n")
