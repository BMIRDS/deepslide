#!/bin/bash
python compress_images.py --input_folder=wsi_6x/train/a --output_folder=wsi_7.5x/train/a --compression_factor=1.25
python compress_images.py --input_folder=wsi_6x/train/s --output_folder=wsi_7.5x/train/s --compression_factor=1.25
python compress_images.py --input_folder=wsi_6x/train/n --output_folder=wsi_7.5x/train/n --compression_factor=1.25
python compress_images.py --input_folder=wsi_6x/val/a --output_folder=wsi_7.5x/val/a --compression_factor=1.25
python compress_images.py --input_folder=wsi_6x/val/s --output_folder=wsi_7.5x/val/s --compression_factor=1.25
python compress_images.py --input_folder=wsi_6x/val/n --output_folder=wsi_7.5x/val/n --compression_factor=1.25
python compress_images.py --input_folder=wsi_6x/test/a --output_folder=wsi_7.5x/test/a --compression_factor=1.25
python compress_images.py --input_folder=wsi_6x/test/s --output_folder=wsi_7.5x/test/s --compression_factor=1.25
python compress_images.py --input_folder=wsi_6x/test/n --output_folder=wsi_7.5x/test/n --compression_factor=1.25
