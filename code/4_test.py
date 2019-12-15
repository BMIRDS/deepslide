import config
from utils_model import get_predictions

# Run the ResNet on the generated patches.
print("\n\n+++++ Running 4_test.py +++++")
print("\n----- Finding validation patch predictions -----")
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
print("----- Finished finding validation patch predictions -----\n")
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
print("----- Finished finding test patch predictions -----\n")
print("+++++ Finished running 4_test.py +++++\n\n")
