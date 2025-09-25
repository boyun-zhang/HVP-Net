CUDA_VISIBLE_DEVICES=4,5,6,7 \
python -m torch.distributed.launch \
--master_port 2502 \
--nproc_per_node=4 \
main_retrieval.py \
--do_train 1 \
--workers 8 \
--n_display 100 \
--epochs 5 \
--lr 1e-4 \
--coef_lr 1e-3 \
--batch_size 32 \
--batch_size_val 32 \
--anno_path MSRVTT \
--video_path MSRVTT/videos \
--datatype msrvtt \
--max_words 24 \
--max_frames 12 \
--video_framerate 1 \
--split_batch 8 \
--output_dir experiments/MSRVTT \
# --resume_from experiments/MSRVTT/2025-09-25_10\:49\:48/checkpoint_epoch_0.pth
