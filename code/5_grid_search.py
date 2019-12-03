from utils_evaluation import grid_search
import config

# Searching over thresholds for filtering noise.
print("+++++ Running 5_grid_search.py +++++")
print("----- Running grid search -----")
grid_search(pred_folder=config.args.preds_val,
            inference_folder=config.args.inference_val)
print("----- Finished running grid search -----")
print("+++++ Finished running 5_grid_search.py +++++")
