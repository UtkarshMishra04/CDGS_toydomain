#!/bin/bash
#SBATCH --job-name=sweep_matrix
#SBATCH --output=logs/sweep_matrix_%A_%a.out
#SBATCH --error=logs/sweep_matrix_%A_%a.err
#SBATCH --partition=overcap
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=128G
#SBATCH --gres=gpu:a40:1
#SBATCH --array=0-9

# Exit on error
set -e

# Define the parameter space
HORIZON_LENGTHS=($(seq 2 100))  # 99 values
RESAMPLING_STEPS=(1 5 10 15 20)  # 5 values

# Total combinations: 99 * 5 = 495
# Split into 10 array jobs, each handling ~50 combinations
TOTAL_COMBINATIONS=495
NUM_ARRAY_JOBS=10
COMBINATIONS_PER_JOB=$((TOTAL_COMBINATIONS / NUM_ARRAY_JOBS))
START_IDX=$((SLURM_ARRAY_TASK_ID * COMBINATIONS_PER_JOB))

# Last job handles any remaining combinations
if [ $SLURM_ARRAY_TASK_ID -eq $((NUM_ARRAY_JOBS - 1)) ]; then
    END_IDX=$TOTAL_COMBINATIONS
else
    END_IDX=$((START_IDX + COMBINATIONS_PER_JOB))
fi

echo "Array job $SLURM_ARRAY_TASK_ID: Running combinations $START_IDX to $((END_IDX - 1))"

# Loop through assigned combinations
for ((i=START_IDX; i<END_IDX; i++)); do
    # Calculate indices
    NUM_RESAMPLING=${#RESAMPLING_STEPS[@]}
    HORIZON_IDX=$((i / NUM_RESAMPLING))
    RESAMPLING_IDX=$((i % NUM_RESAMPLING))

    # Get the actual values
    HORIZON=${HORIZON_LENGTHS[$HORIZON_IDX]}
    RESAMPLING=${RESAMPLING_STEPS[$RESAMPLING_IDX]}

    echo "Running combination $i: horizon=$HORIZON, resampling=$RESAMPLING"

    uv run sampling_time_single.py \
        --horizon-length $HORIZON \
        --enable-pruning False \
        --num-resampling-steps $RESAMPLING \
        --output-directory profile/sweep_horizon_resampling_matrix
done

echo "Array job $SLURM_ARRAY_TASK_ID completed!"
