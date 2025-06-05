# Lightning Settings
# CUDA_VISIBLE_DEVICES=0 \
OMP_NUM_THREADS=16 \
torchrun --nproc_per_node=4 \
train.py \
--strategy fsdp \
"$@"