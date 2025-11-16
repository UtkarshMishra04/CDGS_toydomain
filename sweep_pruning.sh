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
#SBATCH --array=0-4

# Exit on error
set -e

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

# Total pairs: 15 (5+4+3+2+1)
# Split into 5 array jobs, each handling 3 pairs
TOTAL_PAIRS=${#PAIRS[@]}
NUM_ARRAY_JOBS=5
PAIRS_PER_JOB=$((TOTAL_PAIRS / NUM_ARRAY_JOBS))
START_IDX=$((SLURM_ARRAY_TASK_ID * PAIRS_PER_JOB))

# Last job handles any remaining pairs
if [ $SLURM_ARRAY_TASK_ID -eq $((NUM_ARRAY_JOBS - 1)) ]; then
    END_IDX=$TOTAL_PAIRS
else
    END_IDX=$((START_IDX + PAIRS_PER_JOB))
fi

echo "Array job $SLURM_ARRAY_TASK_ID: Running pairs $START_IDX to $((END_IDX - 1))"

# Loop through assigned pairs
for ((i=START_IDX; i<END_IDX; i++)); do
    PAIR=${PAIRS[$i]}
    PRUNING_START=$(echo $PAIR | cut -d',' -f1)
    PRUNING_END=$(echo $PAIR | cut -d',' -f2)

    echo "Running pair $i: pruning_start=$PRUNING_START, pruning_end=$PRUNING_END"

    uv run sampling_time_single.py \
        --horizon-length 10 \
        --num-resampling-steps 5 \
        --enable-pruning True \
        --pruning-start $PRUNING_START \
        --pruning-end $PRUNING_END \
        --output-directory profile/sweep_pruning
done

echo "Array job $SLURM_ARRAY_TASK_ID completed!"
