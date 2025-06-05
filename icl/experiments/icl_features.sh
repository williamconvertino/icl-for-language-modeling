source ~/miniconda3/etc/profile.d/conda.sh
conda activate icl
../scripts/run_experiment.sh --preset icl --n_feature_blocks 1 --n_icl_blocks 3 --use_icl_for_features true