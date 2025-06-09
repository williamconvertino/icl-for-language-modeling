#!/bin/bash
#SBATCH --job-name=icl-language
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64GB
#SBATCH --time=48:00:00
#SBATCH --partition=gpu-common
#SBATCH --output=logs/%x-%j.out

source ~/.bashrc
conda activate icl

# Use the first device, unless specified
DEVICES="0"

for arg in "$@"; do
  if [[ "$arg" == --device=* ]]; then
    DEVICES="${arg#*=}"
  fi
done

export CUDA_VISIBLE_DEVICES=$DEVICES

export OMP_NUM_THREADS=10
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --nproc_per_node=1 ../main.py "$@"