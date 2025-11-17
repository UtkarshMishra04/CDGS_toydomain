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
#SBATCH --array=0-11

# Exit on error
set -e

# Define the parameter space
HORIZON_LENGTHS=(5 10 20 30 40)  # 5 values
RESAMPLING_STEPS=(1 5 10 15 20 25)  # 6 values
DATASETS=(multimodal uniform)  # 2 values
TRAINING_MODES=(separate unified)  # 2 values

# Total combinations: 5 * 6 * 2 * 2 = 120
# Split into 12 array jobs, each handling 10 combinations
TOTAL_COMBINATIONS=120
NUM_ARRAY_JOBS=12
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
    # Calculate indices for 4D parameter space
    NUM_RESAMPLING=${#RESAMPLING_STEPS[@]}
    NUM_DATASETS=${#DATASETS[@]}
    NUM_TRAINING_MODES=${#TRAINING_MODES[@]}

    # Flatten 4D indices: i = ((h * R + r) * D + d) * T + t
    # where R=resampling, D=datasets, T=training_modes
    TEMP=$i
    TRAINING_MODE_IDX=$((TEMP % NUM_TRAINING_MODES))
    TEMP=$((TEMP / NUM_TRAINING_MODES))
    DATASET_IDX=$((TEMP % NUM_DATASETS))
    TEMP=$((TEMP / NUM_DATASETS))
    RESAMPLING_IDX=$((TEMP % NUM_RESAMPLING))
    HORIZON_IDX=$((TEMP / NUM_RESAMPLING))

    # Get the actual values
    HORIZON=${HORIZON_LENGTHS[$HORIZON_IDX]}
    RESAMPLING=${RESAMPLING_STEPS[$RESAMPLING_IDX]}
    DATASET=${DATASETS[$DATASET_IDX]}
    TRAINING_MODE=${TRAINING_MODES[$TRAINING_MODE_IDX]}

    # Set output directory based on dataset and training mode
    OUTPUT_DIR="profile/sweep_horizon_resampling_matrix/${DATASET}_${TRAINING_MODE}"

    echo "Running combination $i: horizon=$HORIZON, resampling=$RESAMPLING, dataset=$DATASET, training_mode=$TRAINING_MODE"

    uv run sampling.py \
        --dataset $DATASET \
        --training-mode $TRAINING_MODE \
        --horizon-length $HORIZON \
        --num-resampling-steps $RESAMPLING \
        --disable-pruning \
        --output-directory $OUTPUT_DIR
done

echo "Array job $SLURM_ARRAY_TASK_ID completed!"
