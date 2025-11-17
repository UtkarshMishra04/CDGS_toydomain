#!/bin/bash
#SBATCH --job-name=sweep_resampling
#SBATCH --output=logs/sweep_resampling_%A.out
#SBATCH --error=logs/sweep_resampling_%A.err
#SBATCH --partition=overcap
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=128G
#SBATCH --gres=gpu:l40s:1

# Exit on error
set -e

# go from 1 to 20
RESAMPLING_STEPS=$(seq 1 20)

for NUM_STEPS in ${RESAMPLING_STEPS[@]}; do
    uv run sampling_time_single.py \
        --horizon-length 20 \
        --num-resampling-steps $NUM_STEPS \
        --output-directory profile/sweep_resampling
done
