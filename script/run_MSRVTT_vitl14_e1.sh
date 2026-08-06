#!/bin/bash
# E1: deeper backbone ViT-L/14 (24 layers), layer selection {1,12,24} -> 0,11,23
# batch size is passed as $1 (default 32); fall back to 16/8 if OOM
PYTHON_ENV=/home/boyun/miniconda3/envs/retrieval/bin
BS=${1:-32}

cd "$(dirname "$0")/.." || exit 1

CUDA_VISIBLE_DEVICES=0 \
$PYTHON_ENV/python main_retrieval.py \
--do_train 1 \
--workers 8 \
--n_display 100 \
--epochs 5 \
--lr 1e-4 \
--coef_lr 1e-3 \
--batch_size $BS \
--batch_size_val 32 \
--anno_path datasets/MSR-VTT/raw_data \
--video_path datasets/MSR-VTT/videos \
--datatype msrvtt \
--max_words 24 \
--max_frames 12 \
--video_framerate 1 \
--split_batch 8 \
--base_encoder ViT-L/14 \
--layer_list 0,11,23 \
--grad_ckpt 1 \
--output_dir experiments/MSRVTT
