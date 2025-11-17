#!/bin/bash
#SBATCH --job-name=sweep_horizon_no_pruning
#SBATCH --output=logs/sweep_horizon_no_pruning_%A.out
#SBATCH --error=logs/sweep_horizon_no_pruning_%A.err
#SBATCH --partition=overcap
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=128G
#SBATCH --gres=gpu:a40:1

# Exit on error
set -e

# go from 2 to 100
HORIZON_LENGTHS=$(seq 2 100)

for HORIZON_LENGTH in ${HORIZON_LENGTHS[@]}; do
    uv run sampling_time_single.py \
        --horizon-length $HORIZON_LENGTH \
        --num-resampling-steps 1 \
        --enable-pruning False \
        --output-directory profile/sweep_horizon_no_pruning
done
