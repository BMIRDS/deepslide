import config
from utils_split import split

print("\n\n+++++ Running 1_split.py +++++")
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
print("+++++ Finished running 1_split.py +++++\n\n")
