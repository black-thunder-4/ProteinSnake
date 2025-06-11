#!/usr/bin/env bash

# Load Conda
source /system/apps/studentenv/miniconda3/etc/profile.d/conda.sh
conda activate /system/apps/studentenv/kranzing/thesis-env

# Go to project root
cd /system/user/studentwork/kranzing/bsc_thesis/openmm
mkdir -p logs

# Define GPU list
GPUS=(0 0 1 1 2 2 3 3 4 4 5 5 6 6 7 7)

for idx in "${!GPUS[@]}"; do
  gpu=${GPUS[$idx]}
  (
    export CUDA_VISIBLE_DEVICES=$gpu
    echo "[$(date '+%H:%M:%S')] Starting job $idx on GPU $gpu"
    python main.py --model egnn \
      --gpu ${gpu} \
      --train_set_path train_trajectories \
      --val_set_path val_trajectories \
      --storage_path optuna.db \
      --study_name egnn \
      --save_dir egnn_models \
      --n_trials 500 \
	  --timeout_per_trial 7200 \
      2>&1 | tee logs/run${idx}_gpu${gpu}.log
  ) &
done

wait
echo "All jobs launched."