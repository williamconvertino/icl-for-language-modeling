# Lightning Settings
CUDA_VISIBLE_DEVICES=1,2,3 \
OMP_NUM_THREADS=16 \
torchrun --nproc_per_node=3 \
train.py \
--strategy fsdp \
"$@"