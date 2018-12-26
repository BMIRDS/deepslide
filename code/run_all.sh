#!/bin/(shell)
python code/1_split.py
python code/2_process_patches.py
CUDA_VISIBLE_DEVICES=0 python code/3_train.py
CUDA_VISIBLE_DEVICES=0 python code/4_test.py
python code/5_grid_search.py
python code/6_visualize.py
python code/7_final_test.py