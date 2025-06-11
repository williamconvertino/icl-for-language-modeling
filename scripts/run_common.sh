#!/bin/bash
#SBATCH --job-name=icl-language
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --partition=common-gpu
#SBATCH --requeue
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -e
source ~/.bashrc
conda activate icl

# Default to using GPU 0 unless overridden
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
