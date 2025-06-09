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

# Setup environment
source ~/.bashrc
source ../setup_env.sh || { echo "Failed to set up environment"; exit 1; }

# Performance vars
export OMP_NUM_THREADS=10
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run the experiment
torchrun --nproc_per_node=1 ../main.py "$@"