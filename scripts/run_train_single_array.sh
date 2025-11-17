#!/bin/bash
#SBATCH --job-name=train_single
#SBATCH --output=logs/train_single_%A_%a.out
#SBATCH --error=logs/train_single_%A_%a.err
#SBATCH --array=0-1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=overcap

# Create logs directory if it doesn't exist
mkdir -p logs

# Define dataset types
DATASET_TYPES=("multimodal" "uniform")
DATASET=${DATASET_TYPES[$SLURM_ARRAY_TASK_ID]}

echo "========================================="
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "Dataset: $DATASET"
echo "Training on concatenated dataset (start + bridge + end)"
echo "========================================="

# Training parameters
NUM_SAMPLES=10000
NUM_EPOCHS=1000
BATCH_SIZE=256
LR=1e-4
SEED=42

# Run training on concatenated dataset
uv run train_single.py \
    --dataset $DATASET \
    --num_samples $NUM_SAMPLES \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --seed $SEED

echo "Training complete for $DATASET concatenated dataset"
