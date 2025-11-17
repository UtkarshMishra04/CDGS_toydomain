#!/bin/bash
#SBATCH --job-name=train_diffusion
#SBATCH --output=logs/train_%A_%a.out
#SBATCH --error=logs/train_%A_%a.err
#SBATCH --array=0-5
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=overcap

# Create logs directory if it doesn't exist
mkdir -p logs

# Define dataset types and components
DATASET_TYPES=("multimodal" "uniform")
COMPONENTS=("start" "bridge" "end")

# Calculate indices
DATASET_IDX=$((SLURM_ARRAY_TASK_ID / 3))
COMPONENT_IDX=$((SLURM_ARRAY_TASK_ID % 3))

DATASET=${DATASET_TYPES[$DATASET_IDX]}
COMPONENT=${COMPONENTS[$COMPONENT_IDX]}

echo "========================================="
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "Dataset: $DATASET"
echo "Component: $COMPONENT"
echo "========================================="

# Training parameters
NUM_SAMPLES=10000
NUM_EPOCHS=1000
BATCH_SIZE=256
LR=1e-4
SEED=42

# Run training
uv run train.py \
    --dataset $DATASET \
    --component $COMPONENT \
    --num_samples $NUM_SAMPLES \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --seed $SEED

echo "Training complete for $DATASET $COMPONENT"
