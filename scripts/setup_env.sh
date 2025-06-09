#!/bin/bash

ENV_NAME="icl"
ENV_FILE="environment.yml"

cd ".."

# Create conda environment if it doesn't exist
if conda env list | grep -qE "^$ENV_NAME\s"; then
    echo "Conda environment '$ENV_NAME' already exists."
else
    echo "Creating conda environment '$ENV_NAME' from $ENV_FILE..."
    conda env create -n "$ENV_NAME" -f "$ENV_FILE"
fi

# Activate conda environment and install src package
echo "Activating conda environment '$ENV_NAME' and installing src package..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME" && pip install -e .

cd "scripts"