#!/bin/bash
#SBATCH --job-name=sweep_pruning
#SBATCH --output=logs/sweep_pruning_%A_%a.out
#SBATCH --error=logs/sweep_pruning_%A_%a.err
#SBATCH --partition=overcap
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=128G
#SBATCH --gres=gpu:a40:1
#SBATCH --array=0-5

# Exit on error
set -e

# Define parameter space
DATASETS=(multimodal uniform)  # 2 values
TRAINING_MODES=(separate unified)  # 2 values

# Generate all (start, end) pairs where end > start, in 0.2 increments
# start: 0.0, 0.2, 0.4, 0.6, 0.8
# end: 0.2, 0.4, 0.6, 0.8, 1.0
PAIRS=()
for start in 0.0 0.2 0.4 0.6 0.8; do
    for end in 0.2 0.4 0.6 0.8 1.0; do
        # Only add if end > start (using awk for float comparison)
        if awk "BEGIN {exit !($end > $start)}"; then
            PAIRS+=("$start,$end")
        fi
    done
done

# Total combinations: 15 pairs * 2 datasets * 2 training_modes = 60
# Split into 6 array jobs, each handling 10 combinations
TOTAL_COMBINATIONS=$((${#PAIRS[@]} * ${#DATASETS[@]} * ${#TRAINING_MODES[@]}))
NUM_ARRAY_JOBS=6
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
    # Calculate indices for 3D parameter space
    # i = (p * D + d) * T + t
    # where p=pair_idx, D=datasets, T=training_modes
    NUM_DATASETS=${#DATASETS[@]}
    NUM_TRAINING_MODES=${#TRAINING_MODES[@]}

    TEMP=$i
    TRAINING_MODE_IDX=$((TEMP % NUM_TRAINING_MODES))
    TEMP=$((TEMP / NUM_TRAINING_MODES))
    DATASET_IDX=$((TEMP % NUM_DATASETS))
    PAIR_IDX=$((TEMP / NUM_DATASETS))

    # Get the actual values
    PAIR=${PAIRS[$PAIR_IDX]}
    PRUNING_START=$(echo $PAIR | cut -d',' -f1)
    PRUNING_END=$(echo $PAIR | cut -d',' -f2)
    DATASET=${DATASETS[$DATASET_IDX]}
    TRAINING_MODE=${TRAINING_MODES[$TRAINING_MODE_IDX]}

    # Set output directory based on dataset and training mode
    OUTPUT_DIR="profile/sweep_pruning/${DATASET}_${TRAINING_MODE}"

    echo "Running combination $i: dataset=$DATASET, training_mode=$TRAINING_MODE, pruning_start=$PRUNING_START, pruning_end=$PRUNING_END"

    uv run sampling.py \
        --dataset $DATASET \
        --training-mode $TRAINING_MODE \
        --batch-size 1000 \
        --num-samples-to-generate 100 \
        --horizon-length 10 \
        --num-resampling-steps 10 \
        --pruning-start $PRUNING_START \
        --pruning-end $PRUNING_END \
        --output-directory $OUTPUT_DIR
done

echo "Array job $SLURM_ARRAY_TASK_ID completed!"
