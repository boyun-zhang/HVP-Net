#!/bin/bash
# E0: local baseline re-run, ViT-B/32 (12 layers), layer selection {1,6,12} -> 0,5,11
PYTHON_ENV=/home/boyun/miniconda3/envs/retrieval/bin

cd "$(dirname "$0")/.." || exit 1

CUDA_VISIBLE_DEVICES=0 \
$PYTHON_ENV/python main_retrieval.py \
--do_train 1 \
--workers 8 \
--n_display 100 \
--epochs 5 \
--lr 1e-4 \
--coef_lr 1e-3 \
--batch_size 32 \
--batch_size_val 32 \
--anno_path datasets/MSR-VTT/raw_data \
--video_path datasets/MSR-VTT/videos \
--datatype msrvtt \
--max_words 24 \
--max_frames 12 \
--video_framerate 1 \
--split_batch 8 \
--base_encoder ViT-B/32 \
--layer_list 0,5,11 \
--output_dir experiments/MSRVTT
