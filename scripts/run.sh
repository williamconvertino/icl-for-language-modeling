#!/bin/bash
#SBATCH --job-name=icl-language
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --partition=scavenger-gpu
#SBATCH --exclude=dcc-youlab-gpu-28,dcc-gehmlab-gpu-56
#SBATCH --nodelist=dcc-allenlab-gpu-[01-04],dcc-allenlab-gpu-[05-12],dcc-majoroslab-gpu-[01-08],dcc-yaolab-gpu-[01-08],dcc-wengerlab-gpu-01,dcc-engelhardlab-gpu-[01-04],dcc-motesa-gpu-[01-04],dcc-pbenfeylab-gpu-[01-04],dcc-vossenlab-gpu-[01-04],dcc-youlab-gpu-[01-56],dcc-mastatlab-gpu-01,dcc-viplab-gpu-01,dcc-youlab-gpu-57
#HIGH VRAM --nodelist=dcc-allenlab-gpu-[01-04],dcc-allenlab-gpu-[05-12],dcc-majoroslab-gpu-[01-08]
#SBATCH --requeue
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -e
source ~/.bashrc
conda activate icl-lm

get_free_port() {
  while : ; do
    PORT=$((29500 + RANDOM % 500))
    (echo >/dev/tcp/127.0.0.1/$PORT) &>/dev/null || break
  done
  echo $PORT
}

MASTER_PORT=$(get_free_port)

DEVICES="0"
STRATEGY="auto"
ARGS=()

for arg in "$@"; do
  if [[ "$arg" == --device=* ]]; then
    DEVICES="${arg#*=}"
  elif [[ "$arg" == --strategy=* ]]; then
    STRATEGY="${arg#*=}"
  elif [[ "$arg" == --job-name=* ]]; then
    # Skip SLURM-specific arguments
    continue
  else
    ARGS+=("$arg")
  fi
done

# Count number of devices
IFS=',' read -ra DEVICE_LIST <<< "$DEVICES"
NUM_DEVICES="${#DEVICE_LIST[@]}"

# Export environment
export CUDA_VISIBLE_DEVICES=$DEVICES
export OMP_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Launching with $NUM_DEVICES processes (1 per device)"

# Run with torchrun
torchrun --master_port=$MASTER_PORT --nproc_per_node=$NUM_DEVICES ../main.py "${ARGS[@]}" --strategy $STRATEGY