#!/bin/bash
#SBATCH --job-name=icl_job
#SBATCH --output=icl_job.out
#SBATCH --error=icl_job.err
#SBATCH --time=01:00:00
#SBATCH --partition=scavenger-gpu
#SBATCH --nodelist=dcc-engelhardlab-gpu-01
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Load Conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate icl

# Run your experiment
/cwork/wac20/icl-for-language-modeling/icl/scripts/run_experiment.sh \
  --preset icl \
  --n_feature_blocks 1 \
  --n_icl_blocks 3 \
  --share_heads_for_icl false