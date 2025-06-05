# Lightning Settings
# CUDA_VISIBLE_DEVICES=0,1,2 \
OMP_NUM_THREADS=16 \
torchrun --nproc_per_node=1 \
train.py \
--strategy auto \
"$@"